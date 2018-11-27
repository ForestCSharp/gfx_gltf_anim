extern crate cimgui;
use self::cimgui::*;

use std::ffi::CString;
use std::fs;
use std::io::{Read};

#[cfg(feature = "dx12")]
extern crate gfx_backend_dx12 as back;
#[cfg(feature = "metal")]
extern crate gfx_backend_metal as back;
#[cfg(feature = "vulkan")]
extern crate gfx_backend_vulkan as back;

use back::Backend as B;

extern crate gfx_hal as hal;
use hal::{PhysicalDevice, Device, Backend, DescriptorPool};
use hal::format::{ AsFormat, Rgba8Unorm as ColorFormat };

extern crate winit;

use ::mesh::GpuBuffer;
use ::gfx_helpers;
use ::glsl_to_spirv;

pub struct CimguiHal {
	gfx_data : CimguiGfxData,
	font_data : CimguiFontData,
}

pub struct CimguiGfxData {
	desc_pool: <B as Backend>::DescriptorPool,
	desc_set : <B as Backend>::DescriptorSet,
	renderpass : <B as Backend>::RenderPass,
	pipeline: <B as Backend>::GraphicsPipeline,
	pipeline_layout: <B as Backend>::PipelineLayout,
	vertex_buffer : Option<GpuBuffer>,
	index_buffer : Option<GpuBuffer>,
}

pub struct CimguiFontData {
	image: <B as Backend>::Image, //Image
	memory: <B as Backend>::Memory, //Memory
	image_view : <B as Backend>::ImageView, //Image View
	sampler : <B as Backend>::Sampler, //Sampler
}

impl CimguiHal {
	pub fn new(device: &back::Device, physical_device : &back::PhysicalDevice, queue_group : &mut hal::QueueGroup<B, hal::Graphics>, color_format : &hal::format::Format, depth_format : &hal::format::Format) -> CimguiHal {

		//Create gfx resources
		let gfx_data = CimguiHal::create_gfx_resources(device, color_format, depth_format);

		unsafe {

			igCreateContext( std::ptr::null_mut());

			let io = igGetIO();

			//Create GPU Objects for Font
			let mut pixels = std::ptr::null_mut();
			let mut width = 0;
			let mut height = 0;
			let mut pixel_size = 0;
			ImFontAtlas_GetTexDataAsRGBA32((*io).Fonts, &mut pixels, &mut width, &mut height, &mut pixel_size);

			let upload_size = (width * height * 4 * std::mem::size_of::<i8>() as i32) as u64;

			let font_image_buffer_unbound = device
				.create_buffer(upload_size, hal::buffer::Usage::TRANSFER_SRC)
				.unwrap();

			let font_buffer_req = device.get_buffer_requirements(&font_image_buffer_unbound);

			let font_buffer_upload_type = gfx_helpers::get_memory_type(&physical_device, &font_buffer_req, hal::memory::Properties::CPU_VISIBLE);

			let font_image_upload_memory = device
				.allocate_memory(font_buffer_upload_type, font_buffer_req.size)
				.unwrap();

			let font_image_upload_buffer = device
				.bind_buffer_memory(&font_image_upload_memory, 0, font_image_buffer_unbound)
				.unwrap();

			// copy image data into staging buffer
			{
				let mut data = device
					.acquire_mapping_writer::<u8>(&font_image_upload_memory, 0..font_buffer_req.size)
					.unwrap();

				data[0..(upload_size as usize)].copy_from_slice(std::slice::from_raw_parts(pixels, upload_size as usize));
				
				device.release_mapping_writer(data).unwrap();
			}

			let kind = hal::image::Kind::D2(width as hal::image::Size, height as hal::image::Size, 1, 1);
			let font_image_unbound = device
				.create_image(
					kind, 
					1, 
					ColorFormat::SELF,
					hal::image::Tiling::Optimal,
					hal::image::Usage::TRANSFER_DST | hal::image::Usage::SAMPLED,
            		hal::image::ViewCapabilities::empty(),
				).unwrap();

			let font_image_reqs = device.get_image_requirements(&font_image_unbound);

			let font_memory_type = gfx_helpers::get_memory_type(&physical_device, &font_image_reqs, hal::memory::Properties::DEVICE_LOCAL);

			let font_image_memory = device.allocate_memory(font_memory_type, font_image_reqs.size).unwrap();

			let font_image = device.bind_image_memory(&font_image_memory, 0, font_image_unbound).unwrap();

			let font_image_view = device.create_image_view(
				&font_image, 
				hal::image::ViewKind::D2,
				ColorFormat::SELF,
				hal::format::Swizzle::NO,
				hal::image::SubresourceRange {
					aspects: hal::format::Aspects::COLOR,
					levels: 0..1,
					layers: 0..1,
				},
			).unwrap();

			let font_sampler = device.create_sampler(hal::image::SamplerInfo::new(
				hal::image::Filter::Linear,
				hal::image::WrapMode::Tile
				)).unwrap();

			//Transfer Font Data from Buffer to Image
			let mut command_pool = device.create_command_pool_typed(queue_group, hal::pool::CommandPoolCreateFlags::TRANSIENT, 1)
                            .expect("Can't create command pool");

			let submit = {
				let mut cmd_buffer = command_pool.acquire_command_buffer(false);

				let color_range = hal::image::SubresourceRange {
					aspects: hal::format::Aspects::COLOR,
					levels: 0..1,
					layers: 0..1,
				};

				let image_barrier = hal::memory::Barrier::Image {
					states: (hal::image::Access::empty(), hal::image::Layout::Undefined)
						..(hal::image::Access::TRANSFER_WRITE, hal::image::Layout::TransferDstOptimal),
					target: &font_image,
					range: color_range.clone(),
				};

				 cmd_buffer.pipeline_barrier(
					hal::pso::PipelineStage::TOP_OF_PIPE..hal::pso::PipelineStage::TRANSFER,
					hal::memory::Dependencies::empty(),
					&[image_barrier],
				);

				cmd_buffer.copy_buffer_to_image(
					&font_image_upload_buffer,
					&font_image,
					hal::image::Layout::TransferDstOptimal,
					&[hal::command::BufferImageCopy {
						buffer_offset: 0,
						buffer_width: width as u32,
						buffer_height: height as u32,
						image_layers: hal::image::SubresourceLayers {
							aspects: hal::format::Aspects::COLOR,
							level: 0,
							layers: 0..1,
						},
						image_offset: hal::image::Offset { x: 0, y: 0, z: 0 },
						image_extent: hal::image::Extent {
							width: width as u32,
							height: height as u32,
							depth: 1,
						},
					}],
				);

				let image_barrier = hal::memory::Barrier::Image {
					states: (hal::image::Access::TRANSFER_WRITE, hal::image::Layout::TransferDstOptimal)
						..(hal::image::Access::SHADER_READ, hal::image::Layout::ShaderReadOnlyOptimal),
					target: &font_image,
					range: color_range.clone(),
				};
				cmd_buffer.pipeline_barrier(
					hal::pso::PipelineStage::TRANSFER..hal::pso::PipelineStage::FRAGMENT_SHADER,
					hal::memory::Dependencies::empty(),
					&[image_barrier],
				);

				cmd_buffer.finish()
			};

			let mut transfer_fence = device.create_fence(false).unwrap();
			let submission = hal::queue::Submission::new().submit(Some(submit));
        	queue_group.queues[0].submit(submission, Some(&mut transfer_fence));
        	device.wait_for_fence(&transfer_fence, !0).expect("Can't wait for fence");

			//Write our font data to our descriptor set
			device.write_descriptor_sets( vec![
				hal::pso::DescriptorSetWrite {
					set: &gfx_data.desc_set,
					binding: 0,
					array_offset: 0,
					descriptors: Some(hal::pso::Descriptor::CombinedImageSampler(&font_image_view, hal::image::Layout::Undefined, &font_sampler)),
				},
			]);

			//Setup Key Mappings
			CimguiHal::set_key_mapping(ImGuiKey__ImGuiKey_Tab, winit::VirtualKeyCode::Tab as i32);
			
			CimguiHal::set_key_mapping(ImGuiKey__ImGuiKey_LeftArrow, winit::VirtualKeyCode::Left as i32);
			CimguiHal::set_key_mapping(ImGuiKey__ImGuiKey_RightArrow, winit::VirtualKeyCode::Right as i32);
			CimguiHal::set_key_mapping(ImGuiKey__ImGuiKey_UpArrow, winit::VirtualKeyCode::Up as i32);
			CimguiHal::set_key_mapping(ImGuiKey__ImGuiKey_DownArrow, winit::VirtualKeyCode::Down as i32);
			
			CimguiHal::set_key_mapping(ImGuiKey__ImGuiKey_PageUp, winit::VirtualKeyCode::PageUp as i32);
			CimguiHal::set_key_mapping(ImGuiKey__ImGuiKey_PageDown, winit::VirtualKeyCode::PageDown as i32);
			CimguiHal::set_key_mapping(ImGuiKey__ImGuiKey_Home, winit::VirtualKeyCode::Home as i32);
			CimguiHal::set_key_mapping(ImGuiKey__ImGuiKey_End, winit::VirtualKeyCode::End as i32);
			CimguiHal::set_key_mapping(ImGuiKey__ImGuiKey_Insert, winit::VirtualKeyCode::Insert as i32);
			CimguiHal::set_key_mapping(ImGuiKey__ImGuiKey_Delete, winit::VirtualKeyCode::Delete as i32);

			CimguiHal::set_key_mapping(ImGuiKey__ImGuiKey_Backspace, winit::VirtualKeyCode::Back as i32);
			CimguiHal::set_key_mapping(ImGuiKey__ImGuiKey_Space, winit::VirtualKeyCode::Space as i32);
			CimguiHal::set_key_mapping(ImGuiKey__ImGuiKey_Enter, winit::VirtualKeyCode::Return as i32);
			CimguiHal::set_key_mapping(ImGuiKey__ImGuiKey_Escape, winit::VirtualKeyCode::Escape as i32);

			CimguiHal::set_key_mapping(ImGuiKey__ImGuiKey_A, winit::VirtualKeyCode::A as i32);
			CimguiHal::set_key_mapping(ImGuiKey__ImGuiKey_C, winit::VirtualKeyCode::C as i32);
			CimguiHal::set_key_mapping(ImGuiKey__ImGuiKey_V, winit::VirtualKeyCode::V as i32);
			CimguiHal::set_key_mapping(ImGuiKey__ImGuiKey_X, winit::VirtualKeyCode::X as i32);
			CimguiHal::set_key_mapping(ImGuiKey__ImGuiKey_Y, winit::VirtualKeyCode::Y as i32);
			CimguiHal::set_key_mapping(ImGuiKey__ImGuiKey_Z, winit::VirtualKeyCode::Z as i32);

			CimguiHal {
				gfx_data : gfx_data,
				font_data : CimguiFontData {
					image  : font_image,
					memory : font_image_memory,
					image_view : font_image_view,
					sampler : font_sampler,
				}
			}
		}
	}

	pub fn update_mouse_state(&mut self, mouse_button_states: [bool;5], mouse_pos : [f32;2], mouse_wheel: [f32;2]) {
		unsafe {
			let io = igGetIO();
			(*io).MouseDown = mouse_button_states;
			(*io).MousePos.x = mouse_pos[0];
			(*io).MousePos.y = mouse_pos[1];
			(*io).MouseWheelH += mouse_wheel[0];
			(*io).MouseWheel += mouse_wheel[1];
		}
	}

	pub fn set_key_mapping(imgui_key : ImGuiKey_, mapping : i32) {
		unsafe {
			let io = igGetIO();
			(*io).KeyMap[imgui_key as usize] = mapping;
		}
	}

	pub fn update_modifier_state(&self, key_ctrl : bool, key_shift : bool, key_alt : bool, key_super : bool) {
		unsafe {
			let io = igGetIO();
			(*io).KeyCtrl  = key_ctrl;
			(*io).KeyShift = key_shift;
			(*io).KeyAlt   = key_alt;
			(*io).KeySuper = key_super;
		}
	}

	pub fn update_key_state(&self, key : usize, pressed : bool) {
		unsafe {
			let io = igGetIO();
			(*io).KeysDown[key] = pressed;
		}
	}

	pub fn add_input_character(&self, c : char)
	{
		unsafe {
			ImGuiIO_AddInputCharacter(igGetIO(), c as ImWchar);
		}
	}

	pub fn render(&mut self, width : f32, height : f32, cmd_buffer : &mut hal::command::CommandBuffer<B, hal::Graphics>, framebuffer: &<B as Backend>::Framebuffer, device : &back::Device, physical_device : &back::PhysicalDevice) {
		unsafe {

			////TODO: Remove Test Gui Code Below
			let io = igGetIO();

			(*io).DisplaySize = ImVec2{ x: width, y: height};
			//TODO: Set this to correct value
			(*io).DeltaTime = 1.0f32 / 60.0f32;

			igNewFrame();

			let mut show_demo_window = false;
			let mut test_float = 1.0;

			igBegin(CString::new("Test Window").unwrap().as_ptr(), &mut true, 0);
			igText(CString::new("Hello, world!").unwrap().as_ptr());
			igSliderFloat(CString::new("test float").unwrap().as_ptr(), &mut test_float, 0.0f32, 1.0f32, std::ptr::null(), 1.0f32);
			igEnd();

			igShowDemoWindow(&mut show_demo_window);

			igEndFrame();

			igRender();
			//TODO: Remove Test Gui Code Above

			if igGetDrawData().is_null() {
				println!("Error: ImDrawData is null");
				return;
			}

			let draw_data = *igGetDrawData();

			let mut in_vertices = Vec::new();
			let mut in_indices = Vec::new();

			let mut cmd_lists = draw_data.CmdLists;
			for _i in 0..draw_data.CmdListsCount {
				let cmd_list = *(*cmd_lists);

				//Copy Vertex Data
				let vertex_data = std::slice::from_raw_parts(cmd_list.VtxBuffer.Data, cmd_list.VtxBuffer.Size as usize);
				in_vertices.extend_from_slice(&vertex_data);

				//Copy Index Data
				let index_data = std::slice::from_raw_parts(cmd_list.IdxBuffer.Data, cmd_list.IdxBuffer.Size as usize);
				in_indices.extend_from_slice(&index_data);

				cmd_lists = cmd_lists.offset(1);
			}

			// Early Exit if nothing to render
			if in_vertices.is_empty() || in_indices.is_empty() {
				return;
			}

			//TODO: Better way to handle buffer recreation
			if self.gfx_data.vertex_buffer.is_some() {
				self.gfx_data.vertex_buffer.as_mut().unwrap().recreate(&in_vertices, device, physical_device);
			} else {
			    self.gfx_data.vertex_buffer = Some(GpuBuffer::new(&in_vertices, hal::buffer::Usage::VERTEX, device, physical_device));
			}

			//TODO: Better way to handle buffer recreation
			if self.gfx_data.index_buffer.is_some() {
				self.gfx_data.index_buffer.as_mut().unwrap().recreate(&in_indices, device, physical_device);
			} else {
			    self.gfx_data.index_buffer = Some(GpuBuffer::new(&in_indices, hal::buffer::Usage::INDEX, device, physical_device));
			}

			cmd_buffer.bind_graphics_pipeline(&self.gfx_data.pipeline);
            cmd_buffer.bind_graphics_descriptor_sets(&self.gfx_data.pipeline_layout, 0, Some(&self.gfx_data.desc_set), &[]); //TODO

			cmd_buffer.bind_vertex_buffers(0, Some((&self.gfx_data.vertex_buffer.as_ref().unwrap().buffer, 0)));
            cmd_buffer.bind_index_buffer(hal::buffer::IndexBufferView {
                buffer: &self.gfx_data.index_buffer.as_ref().unwrap().buffer,
                offset: 0,
                index_type: hal::IndexType::U16, //TODO: check for type of indices
            });

			let viewport = hal::pso::Viewport {
                rect: hal::pso::Rect {
                    x: 0,
                    y: 0,
                    w: draw_data.DisplaySize.x as _,
                    h: draw_data.DisplaySize.y as _,
                },
                depth: 0.0..1.0,
            };

			cmd_buffer.set_viewports(0, &[viewport.clone()]);

            {
                let mut encoder = cmd_buffer.begin_render_pass_inline(
                    &self.gfx_data.renderpass,
                    framebuffer,
                    viewport.rect,
                    &[
                        hal::command::ClearValue::Color(hal::command::ClearColor::Float([1.0, 0.2, 0.2, 1.0,]))
                    ],
                );

				let scale = [(2.0 / draw_data.DisplaySize.x), 2.0 / draw_data.DisplaySize.y];
				let translate = [-1.0 - draw_data.DisplayPos.x * scale[0], -1.0 - draw_data.DisplayPos.y * scale[1] as f32];

				encoder.push_graphics_constants(
					&self.gfx_data.pipeline_layout,
					hal::pso::ShaderStageFlags::VERTEX, 0, 
					&std::mem::transmute::<[f32;4],[u32;4]>([scale[0], scale[1], translate[0], translate[1]]) );

				let display_pos = draw_data.DisplayPos;
				let mut cmd_lists = draw_data.CmdLists;
				let mut vtx_offset = 0;
				let mut idx_offset = 0;
				for _i in 0..draw_data.CmdListsCount {
					let cmd_list = *(*cmd_lists);

					let mut cmds = cmd_list.CmdBuffer.Data;
					for _j in 0..cmd_list.CmdBuffer.Size {
						let cmd = *cmds;

						//TODO: user callback

						let scissor_rect = hal::pso::Rect {
							x : { if (cmd.ClipRect.x - display_pos.x) as i16 > 0 { (cmd.ClipRect.x - display_pos.x) as i16 } else { 0 } },
							y : { if (cmd.ClipRect.y - display_pos.y) as i16 > 0 { (cmd.ClipRect.y - display_pos.y) as i16 } else { 0 } },
							w : (cmd.ClipRect.z - cmd.ClipRect.x) as i16,
							h : (cmd.ClipRect.w - cmd.ClipRect.y) as i16,
						};
						
						//FIXME: causing crashes on vulkan backend
						encoder.set_scissors(0, &[scissor_rect]);

						encoder.draw_indexed(idx_offset..(idx_offset + cmd.ElemCount), vtx_offset, 0..1);

						idx_offset += cmd.ElemCount;
						cmds = cmds.offset(1);
					}
					vtx_offset += cmd_list.VtxBuffer.Size;
					cmd_lists = cmd_lists.offset(1);
				}          	
            }
		}
	}

	pub fn shutdown(self) {
		unsafe {
			igDestroyContext(std::ptr::null_mut());
		}

		//TODO: Cleanup gfx-hal resources (gfx_data, font_data)
	}

	//TODO: make this a member function, so we can better handle resizes
	fn create_gfx_resources(device: &back::Device, color_format : &hal::format::Format, depth_format: &hal::format::Format) -> CimguiGfxData {
		
		//Descriptor Set
		let set_layout = device.create_descriptor_set_layout( 
			&[
				//General Uniform (M,V,P, time)
				hal::pso::DescriptorSetLayoutBinding {
					binding: 0,
					ty: hal::pso::DescriptorType::CombinedImageSampler,
					count: 1,
					stage_flags: hal::pso::ShaderStageFlags::FRAGMENT,
					immutable_samplers: false
				},
			],
			&[],
		).expect("Can't create descriptor set layout");

		let mut desc_pool = device.create_descriptor_pool(
			1,
			&[
				hal::pso::DescriptorRangeDesc {
						ty: hal::pso::DescriptorType::CombinedImageSampler,
						count: 1,
					},
			],
		).expect("Can't create descriptor pool");

		let desc_set = desc_pool.allocate_set(&set_layout).unwrap();

		//Renderpass setup
		let renderpass = {
			let attachment = hal::pass::Attachment {
				format: Some(*color_format),
				samples: 1,
				ops: hal::pass::AttachmentOps::new(
					hal::pass::AttachmentLoadOp::Load,
					hal::pass::AttachmentStoreOp::Store,
				),
				stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
				layouts: hal::image::Layout::ColorAttachmentOptimal..hal::image::Layout::Present,
			};

			let depth_attachment = hal::pass::Attachment {
				format: Some(*depth_format),
				samples: 1,
				ops: hal::pass::AttachmentOps::new(hal::pass::AttachmentLoadOp::Clear, hal::pass::AttachmentStoreOp::DontCare),
				stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
				layouts: hal::image::Layout::DepthStencilAttachmentOptimal .. hal::image::Layout::DepthStencilAttachmentOptimal,
        	};

			let subpass = hal::pass::SubpassDesc {
				colors: &[(0, hal::image::Layout::ColorAttachmentOptimal)],
				depth_stencil: None,
				inputs: &[],
				resolves: &[],
				preserves: &[],
			};

			let dependency = hal::pass::SubpassDependency {
				passes: hal::pass::SubpassRef::External..hal::pass::SubpassRef::Pass(0),
				stages: hal::pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT..hal::pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT,
				accesses: hal::image::Access::empty()..(hal::image::Access::COLOR_ATTACHMENT_READ | hal::image::Access::COLOR_ATTACHMENT_WRITE),
			};

			device.create_render_pass(&[attachment, depth_attachment], &[subpass], &[dependency]).expect("failed to create renderpass")
		};
			
		let new_pipeline_layout = device.create_pipeline_layout(Some(set_layout), &[(hal::pso::ShaderStageFlags::VERTEX, 0..4)]).expect("failed to create pipeline layout");

        let new_pipeline = {
            let vs_module = {
                let glsl = fs::read_to_string("data/shaders/imgui.vert").unwrap();
                let spirv: Vec<u8> = glsl_to_spirv::compile(&glsl, glsl_to_spirv::ShaderType::Vertex)
                    .unwrap()
                    .bytes()
                    .map(|b| b.unwrap())
                    .collect();
                device.create_shader_module(&spirv).unwrap()
            };
            let fs_module = {
                let glsl = fs::read_to_string("data/shaders/imgui.frag").unwrap();
                let spirv: Vec<u8> = glsl_to_spirv::compile(&glsl, glsl_to_spirv::ShaderType::Fragment)
                    .unwrap()
                    .bytes()
                    .map(|b| b.unwrap())
                    .collect();
                device.create_shader_module(&spirv).unwrap()
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
                    main_pass: &renderpass,
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
                    hal::pso::BlendState::On {
						alpha:  hal::pso::BlendOp::Add {
							src: hal::pso::Factor::OneMinusSrcAlpha,
							dst: hal::pso::Factor::Zero,
						},
						color:  hal::pso::BlendOp::Add {
							src: hal::pso::Factor::SrcAlpha,
							dst: hal::pso::Factor::OneMinusSrcAlpha,
						}
					}
                ));

                pipeline_desc.vertex_buffers.push(hal::pso::VertexBufferDesc {
                    binding: 0,
                    stride: std::mem::size_of::<ImDrawVert>() as u32,
                    rate: 0,
                });

                pipeline_desc.attributes.push(hal::pso::AttributeDesc {
                    location: 0,
                    binding: 0,
                    element: hal::pso::Element {
                        format: hal::format::Format::Rg32Float,
                        offset: offset_of!(ImDrawVert, pos) as u32,
                    },
                });

				pipeline_desc.attributes.push(hal::pso::AttributeDesc {
                    location: 1,
                    binding: 0,
                    element: hal::pso::Element {
                        format: hal::format::Format::Rg32Float,
                        offset: offset_of!(ImDrawVert, uv) as u32,
                    },
                });

                pipeline_desc.attributes.push(hal::pso::AttributeDesc {
                    location: 2,
                    binding: 0,
                    element: hal::pso::Element {
                        format: hal::format::Format::Rgba8Unorm,
                        offset: offset_of!(ImDrawVert, col) as u32,
                    },
                });

                pipeline_desc.depth_stencil.depth = hal::pso::DepthTest::Off;
                pipeline_desc.depth_stencil.depth_bounds = false;
                pipeline_desc.depth_stencil.stencil = hal::pso::StencilTest::Off;

                device.create_graphics_pipeline(&pipeline_desc, None)
            };

            device.destroy_shader_module(vs_module);
            device.destroy_shader_module(fs_module);

            pipeline.unwrap()
        };

		CimguiGfxData {
			desc_pool : desc_pool,
			desc_set : desc_set,
			renderpass : renderpass,
			pipeline : new_pipeline,
			pipeline_layout : new_pipeline_layout,
			vertex_buffer : None,
			index_buffer : None,
		}
	}
}