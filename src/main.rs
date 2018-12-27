#![cfg_attr(
    not(any(feature = "vulkan", feature = "dx12", feature = "metal")),
    allow(dead_code, unused_extern_crates, unused_imports)
)]

#[cfg(feature = "dx12")]
extern crate gfx_backend_dx12 as back;
#[cfg(feature = "metal")]
extern crate gfx_backend_metal as back;
#[cfg(feature = "vulkan")]
extern crate gfx_backend_vulkan as back;

use back::Backend as B;

extern crate gfx_hal as hal;

extern crate num;
extern crate glsl_to_spirv;
extern crate winit;
extern crate gltf;
extern crate nalgebra_glm as glm;
extern crate time;

#[macro_use]
extern crate memoffset;

use std::fs;
use std::io::{Read};

use hal::{Instance, Device, PhysicalDevice, DescriptorPool, Surface, Swapchain, QueueFamily};

mod mesh;
use mesh::{GpuBuffer, Vertex};

mod cimgui_hal;
use cimgui_hal::*;
use cimgui_hal::cimgui::*;

mod gfx_helpers;
use gfx_helpers::DeviceState;

mod gltf_loader;
use gltf_loader::*;


#[cfg(any(feature = "vulkan", feature = "dx12", feature = "metal"))]
fn main() {

//TODO: Finer-grained unsafe wraps
unsafe {

    println!("Current Target: {}", env!("TARGET"));

    //Create a window builder
    let window_builder = winit::WindowBuilder::new()
        .with_dimensions(winit::dpi::LogicalSize::new(DIMS.width as f64, DIMS.height as f64))
        .with_title("gfx gltf anim".to_string());

    //Create a winit events loop
    let mut events_loop = winit::EventsLoop::new();

    //Create a window, gfx instance, surface, and enumerate our adapters (GPUs)
    let (_window, _instance, mut adapters, mut surface) = {
        let window = window_builder.build(&events_loop).unwrap();
        let instance = back::Instance::create("gfx gltf anim", 1);
        let surface = instance.create_surface(&window);
        let adapters = instance.enumerate_adapters();
        (window, instance, adapters, surface)
    };

    //Just pick the first GPU we find for now
    let adapter = adapters.remove(0);

	let graphics_queue_family = adapter.queue_families.iter().find(|ref family| family.supports_graphics() ).expect("Failed to find Graphics Queue");
	//TODO: try to get a transfer queue that's different than the graphics queue above (or don't?)
	let _transfer_queue_family = adapter.queue_families.iter().find(|ref family| family.supports_transfer() ).expect("Failed to find Transfer Queue");

	let mut gpu = adapter.physical_device.open(&[(&graphics_queue_family, &[1.0; 1])]).expect("failed to create device and queues");

	let mut device_state = DeviceState {
		device : gpu.device,
		physical_device : adapter.physical_device,
		graphics_queue_group : gpu.queues.take(graphics_queue_family.id()).expect("failed to take graphics queue"),
		//transfer_queue_group : gpu.queues.take(transfer_queue_family.id()).expect("failed to take transfer queue"),
	};

    let mut command_pool = device_state.device.create_command_pool_typed(&device_state.graphics_queue_group, hal::pool::CommandPoolCreateFlags::empty())
                            .expect("Can't create command pool");

	let mut gltf_model = GltfModel::new("data/models/CesiumMan.gltf", &device_state);

    let mut cam_pos = glm::vec3(1.0, 0.0, -0.5);
    let mut cam_forward = glm::vec3(0.,0.,0.,) - cam_pos;
    let mut cam_up = glm::vec3(0., 1., 0.);

    let view_matrix = glm::Mat4::identity();

    let perspective_matrix = glm::perspective(
        DIMS.width as f32 / DIMS.height as f32,
        degrees_to_radians(90.0f32),
        0.001,
        10000.0
    );

    let mut camera_uniform_struct = CameraUniform {
        view_matrix: view_matrix.into(),
        proj_matrix: perspective_matrix.into(),
        model_matrix: glm::Mat4::identity().into(),
        time: 0.0,
    };

    //Uniform Buffer Setup
	let mut uniform_gpu_buffer = GpuBuffer::new(&[camera_uniform_struct], 
												hal::buffer::Usage::UNIFORM, 
												hal::memory::Properties::CPU_VISIBLE, 
												&device_state);

    //Descriptor Set
    let set_layout = device_state.device.create_descriptor_set_layout( 
        &[
            //General Uniform (M,V,P, time)
            hal::pso::DescriptorSetLayoutBinding {
                binding: 0,
                ty: hal::pso::DescriptorType::UniformBuffer,
                count: 1,
                stage_flags: hal::pso::ShaderStageFlags::VERTEX | hal::pso::ShaderStageFlags::FRAGMENT,
                immutable_samplers: false
            },
            //Skeleton
            hal::pso::DescriptorSetLayoutBinding {
                binding: 1,
                ty: hal::pso::DescriptorType::UniformBuffer,
                count: 1,
                stage_flags: hal::pso::ShaderStageFlags::VERTEX,
                immutable_samplers: false
            },
        ],
        &[],
    ).expect("Can't create descriptor set layout");
    
    let mut desc_pool = device_state.device.create_descriptor_pool(
        1,
        &[
            hal::pso::DescriptorRangeDesc {
                    ty: hal::pso::DescriptorType::UniformBuffer,
                    count: 2,
                },
        ],
    ).expect("Can't create descriptor pool");

    let desc_set = desc_pool.allocate_set(&set_layout).unwrap();

    //Descriptor write for our two uniform buffers
    device_state.device.write_descriptor_sets( vec![
        hal::pso::DescriptorSetWrite {
            set: &desc_set,
            binding: 0,
            array_offset: 0,
            descriptors: Some(hal::pso::Descriptor::Buffer(&uniform_gpu_buffer.buffer, None..None)),
        },
        hal::pso::DescriptorSetWrite {
            set: &desc_set,
            binding: 1,
            array_offset: 0,
            descriptors: Some(hal::pso::Descriptor::Buffer(&gltf_model.skeleton.gpu_buffer.buffer, None..None)),
        },
    ]);

    let create_swapchain = |device_state : &DeviceState, surface: &mut <back::Backend as hal::Backend>::Surface| {
        let (capabilities, formats, _present_modes, _composite_alphas) = surface.compatibility(&device_state.physical_device);
        let new_format = formats.map_or(hal::format::Format::Rgba8Srgb, |formats| {
            formats
                .iter()
                .find(|new_format| new_format.base_format().1 == hal::format::ChannelType::Srgb)
                .map(|new_format| *new_format)
                .unwrap_or(formats[0])
        });

        let swap_config = hal::SwapchainConfig::from_caps(&capabilities, new_format, DIMS);
        let new_extent = swap_config.extent.to_extent();
        let (new_swapchain, new_backbuffer) = device_state.device.create_swapchain(surface, swap_config, None).expect("Can't create swapchain");

        (new_swapchain, new_backbuffer, new_format, new_extent)
    };

    //Swapchain
    let (mut swap_chain, backbuffer, mut format, mut extent) = create_swapchain(&device_state, &mut surface );

    //Depth Buffer Setup
    let create_depth_buffer = |device_state : &DeviceState, extent: &hal::image::Extent| {
        let depth_format = hal::format::Format::D32Float;
        let mut depth_image = device_state.device.create_image(
            hal::image::Kind::D2(extent.width as _, extent.height as _, 1, 1),
            1,
            depth_format,
            hal::image::Tiling::Optimal,
            hal::image::Usage::DEPTH_STENCIL_ATTACHMENT,
            hal::image::ViewCapabilities::empty()
        ).unwrap();

        let depth_mem_reqs = device_state.device.get_image_requirements(&depth_image);

		let mem_type = gfx_helpers::get_memory_type(&device_state.physical_device, &depth_mem_reqs, hal::memory::Properties::DEVICE_LOCAL);

        let depth_memory = device_state.device.allocate_memory(mem_type, depth_mem_reqs.size).unwrap();
        device_state.device.bind_image_memory(&depth_memory, 0, &mut depth_image).unwrap();

        let depth_view = device_state.device.create_image_view(
            &depth_image,
            hal::image::ViewKind::D2,
            depth_format,
            hal::format::Swizzle::NO,
            hal::image::SubresourceRange {
            aspects: hal::format::Aspects::DEPTH,
            levels: 0 .. 1,
            layers: 0 .. 1,
        },
        ).unwrap();

        (depth_view, depth_image, depth_memory, depth_format)
    };

    let (mut depth_view, mut _depth_image, mut _depth_memory, mut depth_format) = create_depth_buffer(&device_state, &extent);

    //Renderpass
    let create_renderpass = |device_state: &DeviceState, format : &hal::format::Format, depth_format: &hal::format::Format| {
        let attachment = hal::pass::Attachment {
            format: Some(*format),
            samples: 1,
            ops: hal::pass::AttachmentOps::new(
                hal::pass::AttachmentLoadOp::Clear,
                hal::pass::AttachmentStoreOp::Store,
            ),
            stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
            layouts: hal::image::Layout::Undefined..hal::image::Layout::ColorAttachmentOptimal,
        };

        let depth_attachment = hal::pass::Attachment {
            format: Some(*depth_format),
            samples: 1,
            ops: hal::pass::AttachmentOps::new(hal::pass::AttachmentLoadOp::Clear, hal::pass::AttachmentStoreOp::DontCare),
            stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
            layouts: hal::image::Layout::Undefined .. hal::image::Layout::DepthStencilAttachmentOptimal,
        };

        let subpass = hal::pass::SubpassDesc {
            colors: &[(0, hal::image::Layout::ColorAttachmentOptimal)],
            depth_stencil: Some(&(1, hal::image::Layout::DepthStencilAttachmentOptimal)),
            inputs: &[],
            resolves: &[],
            preserves: &[],
        };

        let dependency = hal::pass::SubpassDependency {
            passes: hal::pass::SubpassRef::External..hal::pass::SubpassRef::Pass(0),
            stages: hal::pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT..hal::pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT,
            accesses: hal::image::Access::empty()
                ..(hal::image::Access::COLOR_ATTACHMENT_READ | hal::image::Access::COLOR_ATTACHMENT_WRITE),
        };

        device_state.device.create_render_pass(&[attachment, depth_attachment], &[subpass], &[dependency]).expect("failed to create renderpass")
    };

    let renderpass = create_renderpass(&device_state, &format, &depth_format);

    let create_framebuffers = |device_state: &DeviceState, backbuffer: hal::Backbuffer<back::Backend>, format: &hal::format::Format, extent: &hal::image::Extent, depth_view: &<back::Backend as hal::Backend>::ImageView, renderpass: &<back::Backend as hal::Backend>::RenderPass| {
        match backbuffer {
            hal::Backbuffer::Images(images) => {
                let pairs = images
                    .into_iter()
                    .map(|image| {
                        let rtv = device_state.device.create_image_view(
                            &image, 
                            hal::image::ViewKind::D2, 
                            *format, 
                            hal::format::Swizzle::NO, 
                            hal::image::SubresourceRange {
                                aspects: hal::format::Aspects::COLOR,
                                levels: 0..1,
                                layers: 0..1,
                            },
                            )
                            .unwrap();
                            (image, rtv)
                    })
                    .collect::<Vec<_>>();
                let fbos = pairs
                    .iter()
                    .map(|&(_, ref rtv)| {
                        device_state.device
                            .create_framebuffer(&renderpass, vec![rtv, &depth_view], *extent)
                            .unwrap()
                    })
                    .collect();
                    (pairs, fbos)
            }
            hal::Backbuffer::Framebuffer(fbo) => (Vec::new(), vec![fbo]),
        }
    };

    let (mut frame_images, mut framebuffers) = create_framebuffers(&device_state, backbuffer, &format, &extent, &depth_view, &renderpass);

    let create_pipeline = |device_state: &DeviceState, renderpass: &<back::Backend as hal::Backend>::RenderPass, set_layout: &<back::Backend as hal::Backend>::DescriptorSetLayout| {
        let new_pipeline_layout = device_state.device.create_pipeline_layout(Some(set_layout), &[(hal::pso::ShaderStageFlags::VERTEX, 0..8)]).expect("failed to create pipeline layout");

        let new_pipeline = {
            let vs_module = {
                let glsl = fs::read_to_string("data/shaders/quad.vert").unwrap();
                let spirv: Vec<u8> = glsl_to_spirv::compile(&glsl, glsl_to_spirv::ShaderType::Vertex)
                    .unwrap()
                    .bytes()
                    .map(|b| b.unwrap())
                    .collect();
                device_state.device.create_shader_module(&spirv).unwrap()
            };
            let fs_module = {
                let glsl = fs::read_to_string("data/shaders/quad.frag").unwrap();
                let spirv: Vec<u8> = glsl_to_spirv::compile(&glsl, glsl_to_spirv::ShaderType::Fragment)
                    .unwrap()
                    .bytes()
                    .map(|b| b.unwrap())
                    .collect();
                device_state.device.create_shader_module(&spirv).unwrap()
            };

            let pipeline = {
                let (vs_entry, fs_entry) = (
                    hal::pso::EntryPoint::<back::Backend> {
                        entry: "main",
                        module: &vs_module,
                        specialization: hal::pso::Specialization::default(),
                    },
                    hal::pso::EntryPoint::<back::Backend> {
                        entry: "main",
                        module: &fs_module,
                        specialization: hal::pso::Specialization::default(),
                    },
                );

                let shader_entries = hal::pso::GraphicsShaderSet {
                    vertex: vs_entry,
                    hull: None,
                    domain: None,
                    geometry: None,
                    fragment: Some(fs_entry),
                };

                let subpass = hal::pass::Subpass {
                    index: 0,
                    main_pass: renderpass,
                };

                let mut pipeline_desc = hal::pso::GraphicsPipelineDesc::new(
                    shader_entries,
                    hal::Primitive::TriangleList,
                    hal::pso::Rasterizer::FILL,
                    &new_pipeline_layout,
                    subpass,
                );
                pipeline_desc.blender.targets.push(hal::pso::ColorBlendDesc(
                    hal::pso::ColorMask::ALL,
                    hal::pso::BlendState::ALPHA,
                ));
                pipeline_desc.vertex_buffers.push(hal::pso::VertexBufferDesc {
                    binding: 0,
                    stride: std::mem::size_of::<Vertex>() as u32,
                    rate: 0,
                });

                pipeline_desc.attributes.push(hal::pso::AttributeDesc {
                    location: 0,
                    binding: 0,
                    element: hal::pso::Element {
                        format: hal::format::Format::Rgb32Float,
                        offset: offset_of!(Vertex, a_pos) as u32,
                    },
                });

                pipeline_desc.attributes.push(hal::pso::AttributeDesc {
                    location: 1,
                    binding: 0,
                    element: hal::pso::Element {
                        format: hal::format::Format::Rgba32Float,
                        offset: offset_of!(Vertex, a_col) as u32,
                    },
                });

                pipeline_desc.attributes.push(hal::pso::AttributeDesc {
                    location: 2,
                    binding: 0,
                    element: hal::pso::Element {
                        format: hal::format::Format::Rg32Float,
                        offset: offset_of!(Vertex, a_uv) as u32,
                    },
                });

                pipeline_desc.attributes.push(hal::pso::AttributeDesc {
                    location: 3,
                    binding: 0,
                    element: hal::pso::Element {
                        format: hal::format::Format::Rgb32Float,
                        offset: offset_of!(Vertex, a_norm) as u32,
                    },
                });

                pipeline_desc.attributes.push(hal::pso::AttributeDesc {
                    location: 4,
                    binding: 0,
                    element: hal::pso::Element {
                        format: hal::format::Format::Rgba32Float,
                        offset: offset_of!(Vertex, a_joint_indices) as u32,
                    },
                });

                pipeline_desc.attributes.push(hal::pso::AttributeDesc {
                    location: 5,
                    binding: 0,
                    element: hal::pso::Element {
                        format: hal::format::Format::Rgba32Float,
                        offset: offset_of!(Vertex, a_joint_weights) as u32,
                    },
                });

                pipeline_desc.depth_stencil.depth = hal::pso::DepthTest::On {
                    fun: hal::pso::Comparison::LessEqual,
                    write: true,
                };
                pipeline_desc.depth_stencil.depth_bounds = false;
                pipeline_desc.depth_stencil.stencil = hal::pso::StencilTest::Off;

                device_state.device.create_graphics_pipeline(&pipeline_desc, None)
            };

            device_state.device.destroy_shader_module(vs_module);
            device_state.device.destroy_shader_module(fs_module);

            pipeline.unwrap()
        };

        (new_pipeline, new_pipeline_layout)
    };

    let (pipeline, pipeline_layout) = create_pipeline(&device_state, &renderpass, &set_layout);

	//initialize cimgui
	let mut cimgui_hal = CimguiHal::new( &mut device_state, &format, &depth_format);

    let mut acquisition_semaphore = device_state.device.create_semaphore().unwrap();
    
    let mut frame_fence = device_state.device.create_fence(false).unwrap();
    let mut running = true;
    let mut needs_resize = true;
    let (mut window_width, mut window_height) = (0u32, 0u32);

    let first_timestamp = timestamp();
    let mut last_time = 0.0f64;

    //TODO: Key Hashmap
    let mut w_state = false;
    let mut s_state = false;
    let mut a_state = false;
    let mut d_state = false;
    let mut e_state = false;
    let mut q_state = false;

    let mut num_frames = 0;

    let mut is_fullscreen = false;

	let mut mouse_pos = [0.0, 0.0];
	let mut mouse_button_states =  [ false, false, false, false, false];
	let mut anim_speed : f32 = 1.0;

    while running {
        num_frames += 1;
        
        let mut mouse_delta = (0.0, 0.0);
		let mut scroll_delta = [0.0, 0.0];

        events_loop.poll_events(|event| {
            if let winit::Event::WindowEvent { event, .. } = event.clone() {
                #[allow(unused_variables)]
                match event {
                    winit::WindowEvent::CloseRequested => running = false,
                    winit::WindowEvent::Resized(dims) => {
                        needs_resize = true;
                        let (new_width, new_height) : (u32, u32) = dims.into();
                        window_width = new_width;
                        window_height = new_height;
                    },
					winit::WindowEvent::ReceivedCharacter(c)=> {
						cimgui_hal.add_input_character(c);
					}
                    winit::WindowEvent::KeyboardInput{ device_id, input } => {
						match input.virtual_keycode {
							Some(keycode) => {
								cimgui_hal.update_key_state(keycode as usize, input.state == winit::ElementState::Pressed);
								cimgui_hal.update_modifier_state( input.modifiers.ctrl, input.modifiers.shift, input.modifiers.alt, input.modifiers.logo);
								match keycode {
									winit::VirtualKeyCode::W   => w_state = input.state == winit::ElementState::Pressed,
									winit::VirtualKeyCode::S   => s_state = input.state == winit::ElementState::Pressed,
									winit::VirtualKeyCode::A   => a_state = input.state == winit::ElementState::Pressed,
									winit::VirtualKeyCode::D   => d_state = input.state == winit::ElementState::Pressed,
									winit::VirtualKeyCode::E   => e_state = input.state == winit::ElementState::Pressed,
									winit::VirtualKeyCode::Q   => q_state = input.state == winit::ElementState::Pressed,
									winit::VirtualKeyCode::F11 => {
										if input.state == winit::ElementState::Pressed {
											is_fullscreen = !is_fullscreen;
											if is_fullscreen {
												_window.set_fullscreen(Some(_window.get_primary_monitor()));
											}
											else {
												_window.set_fullscreen(None);
											}
										}
									},
									winit::VirtualKeyCode::Escape => running = false,
									_ => {},
									}
								}
							None => {},
						}
                    },
                    winit::WindowEvent::MouseInput { state, button, ..} => {
						let pressed = state == winit::ElementState::Pressed;
						match button {
							winit::MouseButton::Left   => mouse_button_states[0] = pressed,
							winit::MouseButton::Right  => mouse_button_states[1] = pressed,
							winit::MouseButton::Middle => mouse_button_states[2] = pressed,
							winit::MouseButton::Other(idx) => {
								if (idx as usize) < mouse_button_states.len() {
									mouse_button_states[idx as usize] = pressed;
								}
							}
						}
                    },
					winit::WindowEvent::CursorMoved { position, modifiers, ..} => {
						mouse_pos = [position.x as f32, position.y as f32];
					},
                    _ => (),
                }
            }
            if let winit::Event::DeviceEvent { event, .. }= event.clone() {
                match event {
                    winit::DeviceEvent::MouseMotion { delta } => {
                    	mouse_delta = delta;
                    },
					winit::DeviceEvent::MouseWheel { delta } => {
						
						scroll_delta = match delta {
							winit::MouseScrollDelta::LineDelta(x,y) => [x, y],
							winit::MouseScrollDelta::PixelDelta(pos) =>[pos.x as f32, pos.y as f32],
						};
					}
                    _ => {},
                }
            }
        });

		//FIXME: Mouse Scroll is bad on touch pads
		cimgui_hal.update_mouse_state(mouse_button_states, mouse_pos, scroll_delta);

        let mut forward = 0.0;
        let mut right = 0.0;
        let mut up = 0.0;

        if w_state {
            forward += 1.0;
        }
        if s_state {
            forward -= 1.0;
        }
        if d_state { 
            right += 1.0;
        }
        if a_state {
            right -= 1.0;
        }
        if e_state {
            up += 1.0;
        }
        if q_state {
            up -= 1.0;
        }

        if window_width == 0 || window_height == 0 {
            continue;
        }
        
        if needs_resize {
            device_state.device.wait_idle().unwrap();

            //Destroy old resources
            for framebuffer in framebuffers {
                device_state.device.destroy_framebuffer(framebuffer);
            }

            for (_, rtv) in frame_images {
                device_state.device.destroy_image_view(rtv);
            }

            device_state.device.destroy_swapchain(swap_chain);

            //Build new resources         
            let (new_swapchain, new_backbuffer, new_format, new_extent) = create_swapchain(&device_state, &mut surface );

            swap_chain = new_swapchain;
            format = new_format;
            extent = new_extent;

            let (new_depth_view, new_depth_image, new_depth_memory, new_depth_format) = create_depth_buffer(&device_state, &extent);
            depth_view = new_depth_view;
            _depth_image = new_depth_image;
            _depth_memory = new_depth_memory;
            depth_format = new_depth_format;

            let (new_frame_images, new_framebuffers) = create_framebuffers(&device_state, new_backbuffer, &format, &extent, &depth_view, &renderpass);
            frame_images = new_frame_images;
            framebuffers = new_framebuffers;

            //Update Projection matrix (possible change in aspect ratio)
            camera_uniform_struct.proj_matrix = glm::perspective(
                extent.width as f32 / extent.height as f32,
                degrees_to_radians(90.0f32),
                0.001,
                10000.0
            ).into();
            
            //flipping this to make y point up
            camera_uniform_struct.proj_matrix[1][1] *= -1.0;

            needs_resize = false;
        }

        let time = timestamp() - first_timestamp;
        let delta_time = time - last_time;

        //Rotate cam_forward & cam_up when right mouse pressed using mouse delta
        if mouse_button_states[1] {
            let yaw_rotation = glm::quat_angle_axis(degrees_to_radians(-mouse_delta.0 * 50.0 * delta_time) as f32, &glm::vec3(0.,1.,0.));
            let pitch_rotation = glm::quat_angle_axis(degrees_to_radians(-mouse_delta.1 * 50.0 * delta_time) as f32, &cam_forward.cross(&cam_up));

            let total_rotation = yaw_rotation * pitch_rotation;

            cam_forward = glm::quat_rotate_vec3(&total_rotation, &cam_forward);
            cam_up = glm::quat_rotate_vec3(&total_rotation, &cam_up);
        }

        let move_speed = 3.0;
        let forward_vec = cam_forward * forward * move_speed * delta_time as f32;
        let right_vec =  cam_forward.cross(&cam_up) * right * move_speed * delta_time as f32;
        let up_vec = cam_up * up * move_speed * delta_time as f32;
        let move_vec = forward_vec + right_vec + up_vec;

		if !cimgui_hal.wants_capture_keyboard() {
        	cam_pos += move_vec;
		}

        camera_uniform_struct.view_matrix = glm::look_at(
            &cam_pos,
            &(cam_pos + cam_forward),
            &cam_up
        ).into();

        camera_uniform_struct.model_matrix = (glm::translation(&glm::vec3(0.,0., -1.))
                                              * glm::rotation(std::f32::consts::PI / 4.0, &glm::vec3(0.,1.,0.))).into();

        camera_uniform_struct.time = time as f32;

		uniform_gpu_buffer.reupload(&[camera_uniform_struct], &device_state);

        //Animate Bones
        gltf_model.animate(delta_time *  anim_speed as f64);

		//Upload Bones to GPU
		gltf_model.upload_bones(&device_state);
        
        device_state.device.reset_fence(&frame_fence).unwrap();
        command_pool.reset();

        let frame: hal::SwapImageIndex = {
            match swap_chain.acquire_image(!0, hal::FrameSync::Semaphore(&mut acquisition_semaphore)) {
                Ok(i) => i,
                Err(_) => {
                    needs_resize = true;
                    continue;
                }
            }
        };


		let mut cmd_buffer = command_pool.acquire_command_buffer();

		let viewport = hal::pso::Viewport {
			rect: hal::pso::Rect {
				x: 0,
				y: 0,
				w: extent.width as _,
				h: extent.height as _,
			},
			depth: 0.0..1.0,
		};

		cmd_buffer.set_viewports(0, &[viewport.clone()]);
		cmd_buffer.set_scissors(0, &[viewport.rect]);
		cmd_buffer.bind_graphics_pipeline(&pipeline);
		cmd_buffer.bind_graphics_descriptor_sets(&pipeline_layout, 0, Some(&desc_set), &[]); //TODO

		{
			let mut encoder = cmd_buffer.begin_render_pass_inline(
				&renderpass,
				&framebuffers[frame as usize],
				viewport.rect,
				&[
					hal::command::ClearValue::Color(hal::command::ClearColor::Float([0.2, 0.2, 0.2, 0.0,])),
					hal::command::ClearValue::DepthStencil(hal::command::ClearDepthStencil(1.0, 0))
				],
			);

			gltf_model.record_draw_commands(&mut encoder, 100);
		}

		cimgui_hal.new_frame(window_width as f32, window_height as f32, delta_time as f32);

		//TODO: Safe API for cimgui
		#[allow(unused_unsafe)]
		unsafe {
			use std::ffi::CString;

			igBegin(CString::new("Test Window").unwrap().as_ptr(), &mut true, 0);
			igText(CString::new("Hello, world!").unwrap().as_ptr());
			igSliderFloat(CString::new("Anim Speed").unwrap().as_ptr(), &mut anim_speed, 0.0f32, 20.0f32, std::ptr::null(), 2.0f32);
			igEnd();

			igShowDemoWindow(&mut true);
		}

		cimgui_hal.render(&mut cmd_buffer, &framebuffers[frame as usize], &device_state);

		cmd_buffer.finish();

        let submission_semaphore = device_state.device.create_semaphore().unwrap();

        {
			let submission = hal::queue::Submission {
                command_buffers: Some(&cmd_buffer),
                wait_semaphores: Some((&acquisition_semaphore, hal::pso::PipelineStage::BOTTOM_OF_PIPE)),
                signal_semaphores: Some(&submission_semaphore),
            };
            device_state.graphics_queue_group.queues[0].submit(submission, Some(&mut frame_fence));
        }

        //TODO: Remove once submission_semaphore is working properly
        device_state.device.wait_for_fence(&frame_fence, !0).unwrap();

        // present frame
        if let Err(_) = swap_chain.present(&mut device_state.graphics_queue_group.queues[0], frame, &[submission_semaphore]) {
            needs_resize = true;
        }

        last_time = time;
    }

    let total_time = timestamp() - first_timestamp;
    println!("Avg Frame Time: {}", total_time / num_frames as f64);

    gltf_model.destroy(&device_state);

	cimgui_hal.shutdown(&device_state);
	}
}

#[cfg_attr(rustfmt, rustfmt_skip)]
const DIMS: hal::window::Extent2D = hal::window::Extent2D { width: 1280, height: 720 };

#[derive(Debug, Clone, Copy)]
struct CameraUniform {
    view_matrix:  [[f32;4];4],
    proj_matrix:  [[f32;4];4],
    model_matrix: [[f32;4];4],
    time: f32,
}



fn timestamp() -> f64 {
    let timespec = time::get_time();
    timespec.sec as f64 + (timespec.nsec as f64 / 1000.0 / 1000.0 / 1000.0)
}

fn degrees_to_radians<T>( deg: T) -> T 
where T: num::Float {
    deg * num::cast(0.0174533).unwrap()
}