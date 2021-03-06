#![cfg_attr(
    not(any(feature = "vulkan", feature = "dx12", feature = "metal")),
    allow(dead_code, unused_extern_crates, unused_imports)
)]

#[cfg(feature = "dx11")]
extern crate gfx_backend_dx11 as back;
#[cfg(feature = "dx12")]
extern crate gfx_backend_dx12 as back;
#[cfg(feature = "metal")]
extern crate gfx_backend_metal as back;
#[cfg(feature = "vulkan")]
extern crate gfx_backend_vulkan as back;

use back::Backend as B;

extern crate gfx_hal as hal;

//TODO: switch to shaderc-rs
extern crate glsl_to_spirv;

extern crate num;
extern crate winit;
extern crate nalgebra_glm as glm;
extern crate time;

use std::sync::{Arc, Mutex};
extern crate num_cpus;
extern crate scoped_threadpool;
use scoped_threadpool::Pool;
use std::sync::mpsc::channel;

#[macro_use]
extern crate memoffset;

use std::fs;
use std::collections::HashMap;

use hal::{Instance, Device, PhysicalDevice, DescriptorPool, Surface, Swapchain, QueueFamily, Backend};

mod gpu_buffer;
use gpu_buffer::GpuBuffer;

mod cimgui_hal;
use cimgui_hal::*;
use cimgui_hal::cimgui::*;

mod gfx_helpers;
use gfx_helpers::DeviceState;

mod gltf_loader;
use gltf_loader::*;

mod compute;
use compute::ComputeContext;


#[cfg(any(feature = "vulkan", feature = "dx11", feature = "dx12", feature = "metal"))]
fn main() {

    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Error)
        .init();

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
    let (window, _instance, mut adapters, mut surface) = {
        let window = window_builder.build(&events_loop).unwrap();
        let instance = back::Instance::create("gfx gltf anim", 1);
        let surface = instance.create_surface(&window);
        let adapters = instance.enumerate_adapters();
        (window, instance, adapters, surface)
    };

    //Just pick the first GPU we find for now
    let adapter = adapters.remove(0);
    println!("GPU Name: {}", adapter.info.name);

	//TODO: Query for specific features?
	let features = adapter.physical_device.features();
    //let limits = adapter.physical_device.limits();
    //println!("Limits: {:?}", limits);

    for queue_family in adapter.queue_families.iter() {
        println!("{:?}", queue_family);
    }

	let general_queue_family = adapter.queue_families.iter().find(|family| 
        family.supports_graphics()
        && family.supports_transfer()
        && surface.supports_queue_family(family)
    ).expect("Failed to find Graphics Queue");

    //try to get a dedicated compute queue (or fallback to general queue family)
    let compute_queue_family  = match adapter.queue_families.iter().find(|family| 
        family.supports_compute() 
        && family.id() != general_queue_family.id() 
    ) {
        Some(queue_family) => queue_family,
        None => general_queue_family,
    };

	let mut gpu = adapter.physical_device.open(&[(&general_queue_family, &[1.0; 1]), 
                                                 (&compute_queue_family,  &[1.0; 1])],
                                                features)
                                                .expect("failed to create device and queues");

	let device_state = DeviceState {
		device : gpu.device,
		physical_device : adapter.physical_device,
	};

    let mut general_queue_group = gpu.queues.take::<hal::General>(general_queue_family.id()).expect("failed to take graphics queue");
    let mut compute_queue_group = gpu.queues.take::<hal::Compute>(compute_queue_family.id()).expect("failed to take compute queue");

    let mut command_pool = device_state.device.create_command_pool_typed::<hal::General>(&general_queue_group, hal::pool::CommandPoolCreateFlags::empty())
                            .expect("Can't create command pool");

	let mut gltf_model = GltfModel::new("data/models/Running.glb", &device_state, &mut general_queue_group);

        //Depth Buffer Setup
    let create_depth_buffer = |device_state : &DeviceState, extent: &hal::image::Extent, sampled : bool| {
        let depth_format = hal::format::Format::D32Sfloat;
        let mut depth_image = device_state.device.create_image(
            hal::image::Kind::D2(extent.width as _, extent.height as _, 1, 1),
            1,
            depth_format,
            hal::image::Tiling::Optimal,
            if sampled { 
                hal::image::Usage::DEPTH_STENCIL_ATTACHMENT | hal::image::Usage::SAMPLED 
            } else {
                hal::image::Usage::DEPTH_STENCIL_ATTACHMENT
            },
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
            levels: 0..1,
            layers: 0..1,
        },
        ).unwrap();

        (depth_view, depth_image, depth_memory, depth_format)
    };

    //BEGIN SHADOW MAPPING
    //TODO: also add to quad.frag shader so lighting and shadows come from same source

    let shadow_map_extent = hal::image::Extent {
        width:  2048,
        height: 2048,
        depth: 1,
    };

    let light_pos = glm::vec3(0.0, 4000.0, 0.0);
    let light_dir = glm::vec3(0.0, -1.0, 0.0);
    let light_target = light_pos + light_dir * 1000.0;

    //let light_proj_matrix = glm::perspective_zo(shadow_map_extent.width as f32 / shadow_map_extent.height as f32, degrees_to_radians(30.0f32),5000.0,10.0);
    let light_proj_matrix = glm::ortho_zo(-1000.0, 1000.0, -1000.0, 1000.0, 10000.0, -10000.0);
    //TODO: Ensure "up" is orthogonal to vector formed by "eye" and "center"
    let light_view_matrix = glm::look_at(&light_pos, &light_target, &glm::vec3(1.0, 0.0, 0.0));
    let light_matrix = light_proj_matrix * light_view_matrix;

    let mut shadow_uniform_struct = ShadowUniform {
        shadow_mvp : light_matrix.into(),
        light_dir  : light_dir.into(),
        bias : 0.001,
    };
    //Make y point up
    shadow_uniform_struct.shadow_mvp[1][1] *= -1.0;
    
    let mut shadow_uniform_buffer = GpuBuffer::new( &[shadow_uniform_struct],
                                                    hal::buffer::Usage::UNIFORM,
                                                    hal::memory::Properties::CPU_VISIBLE,
                                                    &device_state,
                                                    &mut general_queue_group);

    let (shadow_depth_view, _shadow_depth_image, _shadow_depth_memory, shadow_depth_format) = create_depth_buffer(&device_state, &shadow_map_extent, true);

    let shadow_sampler = device_state.device.create_sampler(hal::image::SamplerInfo::new(hal::image::Filter::Linear, hal::image::WrapMode::Clamp))
        .expect("Can't create sampler");

    let (_shadow_desc_pool, shadow_desc_set, shadow_renderpass, shadow_pipeline_layout, shadow_pipeline) = {
        let shadow_renderpass = {
            let depth_attachment = hal::pass::Attachment {
                format: Some(shadow_depth_format),
                samples: 1,
                ops: hal::pass::AttachmentOps::new(hal::pass::AttachmentLoadOp::Clear, hal::pass::AttachmentStoreOp::Store),
                stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
                layouts: hal::image::Layout::Undefined..hal::image::Layout::DepthStencilAttachmentOptimal, //TODO: Make Read Only?
            };

            let subpass = hal::pass::SubpassDesc {
                colors: &[],
                depth_stencil: Some(&(0, hal::image::Layout::DepthStencilAttachmentOptimal)),
                inputs: &[],
                resolves: &[],
                preserves: &[],
            };

            device_state.device.create_render_pass(&[depth_attachment], &[subpass], &[]).expect("failed to create renderpass")
        };

        let shadow_set_layout = device_state.device.create_descriptor_set_layout( 
            &[
                //Shadow Uniform (Just one MVP matrix)
                hal::pso::DescriptorSetLayoutBinding {
                    binding: 0,
                    ty: hal::pso::DescriptorType::UniformBuffer,
                    count: 1,
                    stage_flags:  hal::pso::ShaderStageFlags::VERTEX,
                    immutable_samplers: false
                },
            ],
            &[],
        ).expect("Can't create descriptor set layout");

        let mut shadow_desc_pool = device_state.device.create_descriptor_pool(
            1,
            &[
                hal::pso::DescriptorRangeDesc {
                        ty: hal::pso::DescriptorType::UniformBuffer,
                        count: 1,
                    },
            ],
            hal::pso::DescriptorPoolCreateFlags::empty()
        ).expect("Can't create descriptor pool");

        let shadow_desc_set = shadow_desc_pool.allocate_set(&shadow_set_layout).unwrap();

        device_state.device.write_descriptor_sets( vec![
            hal::pso::DescriptorSetWrite {
                set: &shadow_desc_set,
                binding: 0,
                array_offset: 0,
                descriptors: Some(hal::pso::Descriptor::Buffer(&shadow_uniform_buffer.buffer, None..None)),
        }]);

        let shadow_pipeline_layout = device_state.device
            .create_pipeline_layout(Some(shadow_set_layout), &[])
            .expect("failed to create pipeline layout");

        let shadow_pipeline = {
            let vs_module = {
                let glsl = fs::read_to_string("data/shaders/shadow.vert").unwrap();
                let file = glsl_to_spirv::compile(&glsl, glsl_to_spirv::ShaderType::Vertex)
                    .unwrap();
                let spirv: Vec<u32> = hal::read_spirv(file).unwrap();
                device_state.device.create_shader_module(&spirv).unwrap()
            };

            let pipeline = {
                let vs_entry = hal::pso::EntryPoint::<B> {
                        entry: "main",
                        module: &vs_module,
                        specialization: hal::pso::Specialization::default(),
                    };

                let shader_entries = hal::pso::GraphicsShaderSet {
                    vertex: vs_entry,
                    hull: None,
                    domain: None,
                    geometry: None,
                    fragment: None,
                };

                let subpass = hal::pass::Subpass {
                    index: 0,
                    main_pass: &shadow_renderpass,
                };

                let mut pipeline_desc = hal::pso::GraphicsPipelineDesc::new(
                    shader_entries,
                    hal::Primitive::TriangleList,
                    hal::pso::Rasterizer {
                        polygon_mode: hal::pso::PolygonMode::Fill,
                        cull_face: hal::pso::Face::NONE,
                        front_face: hal::pso::FrontFace::CounterClockwise,
                        depth_clamping: false,
                        depth_bias: None,
                        conservative: false,
                    },
                    &shadow_pipeline_layout,
                    subpass,
                );
                pipeline_desc.blender.targets.push(hal::pso::ColorBlendDesc{
                    mask: hal::pso::ColorMask::ALL,
                    blend: Some(hal::pso::BlendState::ALPHA)
                });
                // (
                //     ,
                //     hal::pso::BlendState::ALPHA,
                // ));

                pipeline_desc.vertex_buffers.push(hal::pso::VertexBufferDesc {
                    binding: 0,
                    stride: std::mem::size_of::<Vertex>() as u32,
                    rate: hal::pso::VertexInputRate::Vertex,
                });

                //FIXME: getting STATUS_ILLEGAL_INSTRUCTION here?
                pipeline_desc.attributes.push(hal::pso::AttributeDesc {
                    location: 0,
                    binding: 0,
                    element: hal::pso::Element {
                        format: hal::format::Format::Rgb32Sfloat,
                        offset: offset_of!(Vertex, a_pos) as u32,
                    },
                });

                pipeline_desc.depth_stencil.depth = Some(hal::pso::DepthTest {
                    fun: hal::pso::Comparison::GreaterEqual,
                    write: true,
                });
                pipeline_desc.depth_stencil.depth_bounds = false;
                pipeline_desc.depth_stencil.stencil = None;

                device_state.device.create_graphics_pipeline(&pipeline_desc, None)
            };

            device_state.device.destroy_shader_module(vs_module);

            pipeline.unwrap()
        };

        (shadow_desc_pool, shadow_desc_set, shadow_renderpass, shadow_pipeline_layout, shadow_pipeline)
    };

    //FIXME: one framebuffer per frame? or not because depth-only?
    let shadow_framebuffer = device_state.device
                                .create_framebuffer(&shadow_renderpass, vec![&shadow_depth_view], shadow_map_extent)
                                .unwrap();
    //END SHADOW MAPPING

    let mut cam_pos = glm::vec3(1.0, 0.0, -0.5);
    let mut cam_forward = glm::vec3(0.,0.,0.,) - cam_pos;
    let mut cam_up = glm::vec3(0., 1., 0.);

    let view_matrix = glm::Mat4::identity();

    let perspective_matrix = glm::perspective_zo(
        DIMS.width as f32 / DIMS.height as f32,
        degrees_to_radians(90.0f32),
        100000.0,
        0.01,
    );

    let mut general_uniform_struct = UniformStruct {
        view_matrix: view_matrix.into(),
        proj_matrix: perspective_matrix.into(),
        model_matrix: glm::Mat4::identity().into(),
        time: 0.0,
        pn_triangles_strength: 0.0,
        tess_level: 1.0,
    };

    //Uniform Buffer Setup
	let mut uniform_gpu_buffer = GpuBuffer::new(&[general_uniform_struct], 
												hal::buffer::Usage::UNIFORM, 
												hal::memory::Properties::CPU_VISIBLE, 
												&device_state,
                                                &mut general_queue_group);

    //Descriptor Set
    //FIXME: make this work with models that don't have skeletons
    let set_layout = device_state.device.create_descriptor_set_layout( 
        &[
            //General Uniform (M,V,P, time)
            hal::pso::DescriptorSetLayoutBinding {
                binding: 0,
                ty: hal::pso::DescriptorType::UniformBuffer,
                count: 1,
                stage_flags:  hal::pso::ShaderStageFlags::VERTEX
                            | hal::pso::ShaderStageFlags::HULL
                            | hal::pso::ShaderStageFlags::DOMAIN 
                            | hal::pso::ShaderStageFlags::FRAGMENT,
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
            //Shadow Matrix
            hal::pso::DescriptorSetLayoutBinding {
                binding: 2,
                ty: hal::pso::DescriptorType::UniformBuffer,
                count: 1,
                stage_flags: hal::pso::ShaderStageFlags::VERTEX | hal::pso::ShaderStageFlags::DOMAIN | hal::pso::ShaderStageFlags::FRAGMENT,
                immutable_samplers: false
            },
            //Shadow Map
            hal::pso::DescriptorSetLayoutBinding {
                binding: 3,
                ty: hal::pso::DescriptorType::CombinedImageSampler,
                count: 1,
                stage_flags: hal::pso::ShaderStageFlags::FRAGMENT,
                immutable_samplers: false
            }
        ],
        &[],
    ).expect("Can't create descriptor set layout");
    
    let mut desc_pool = device_state.device.create_descriptor_pool(
        1,
        &[
            hal::pso::DescriptorRangeDesc {
                    ty: hal::pso::DescriptorType::UniformBuffer,
                    count: 3,
                },
            hal::pso::DescriptorRangeDesc {
                    ty: hal::pso::DescriptorType::CombinedImageSampler,
                    count: 1,
                },
        ],
        hal::pso::DescriptorPoolCreateFlags::empty()
    ).expect("Can't create descriptor pool");

    let desc_set = desc_pool.allocate_set(&set_layout).unwrap();

    //Descriptor write for our two uniform buffers (TODO: Handle in gltf_loader)
    //FIXME: make this work with models that don't have skeletons
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
            descriptors: Some(hal::pso::Descriptor::Buffer(&gltf_model.skeletons[0].gpu_buffer.buffer, None..None)),
        },
        hal::pso::DescriptorSetWrite {
            set: &desc_set,
            binding: 2,
            array_offset: 0,
            descriptors: Some(hal::pso::Descriptor::Buffer(&shadow_uniform_buffer.buffer, None..None)),
        },
        hal::pso::DescriptorSetWrite {
            set: &desc_set,
            binding: 3,
            array_offset: 0,
            descriptors: Some(hal::pso::Descriptor::CombinedImageSampler(&shadow_depth_view, hal::image::Layout::DepthStencilReadOnlyOptimal, &shadow_sampler)),
        }
    ]);

    let create_swapchain = |device_state : &DeviceState, surface: &mut <B as hal::Backend>::Surface| {
        let (capabilities, formats, _present_modes) = surface.compatibility(&device_state.physical_device);
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
    let (mut swapchain, backbuffer, mut format, mut extent) = create_swapchain(&device_state, &mut surface);

    let (mut depth_view, mut depth_image, mut depth_memory, mut depth_format) = create_depth_buffer(&device_state, &extent, false);

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
            layouts: hal::image::Layout::Undefined..hal::image::Layout::DepthStencilAttachmentOptimal,
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

    let create_framebuffers = |device_state: &DeviceState, backbuffer: Vec<<B as Backend>::Image>, format: &hal::format::Format, extent: &hal::image::Extent, depth_view: &<B as hal::Backend>::ImageView, renderpass: &<B as hal::Backend>::RenderPass| {
        let extent = hal::image::Extent {
            width: extent.width as _,
            height: extent.height as _,
            depth: 1,
        };
        let pairs = backbuffer
            .into_iter()
            .map(|image| {
                let rtv = device_state.device
                    .create_image_view(
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
                    .create_framebuffer(
                        renderpass,
                        vec![rtv, &depth_view],
                        extent,
                    )
                    .unwrap()
            })
            .collect();
        (pairs, fbos)
    };

    let (mut frame_images, mut framebuffers) : (_, Vec<<B as Backend>::Framebuffer>) = create_framebuffers(&device_state, backbuffer, &format, &extent, &depth_view, &renderpass);

    let create_pipeline = |device_state: &DeviceState, renderpass: &<B as hal::Backend>::RenderPass, set_layout: &<B as hal::Backend>::DescriptorSetLayout, use_wireframe : bool, use_tessellation : bool| {
        
        //TODO: DX12 Doesn't play nice with tessellation at the moment
        let use_tessellation = use_tessellation && !cfg!(feature = "dx12") && !cfg!(feature = "dx11");

        let new_pipeline_layout = device_state.device.create_pipeline_layout(Some(set_layout), &[]).expect("failed to create pipeline layout");

        let new_pipeline = {
            let vs_module = {
                let glsl = fs::read_to_string("data/shaders/quad.vert").unwrap();
                let file = glsl_to_spirv::compile(&glsl, glsl_to_spirv::ShaderType::Vertex)
                    .unwrap();
                let spirv: Vec<u32> = hal::read_spirv(file).unwrap();
                device_state.device.create_shader_module(&spirv).unwrap()
            };

            let tesc_module = if use_tessellation {
                let glsl = fs::read_to_string("data/shaders/pntriangles.tesc").unwrap();
                let file = glsl_to_spirv::compile(&glsl, glsl_to_spirv::ShaderType::TessellationControl)
                    .unwrap();
                let spirv: Vec<u32> = hal::read_spirv(file).unwrap();
                Some(device_state.device.create_shader_module(&spirv).unwrap())
            } 
            else { 
                None 
            };

            let tese_module = if use_tessellation {
                let glsl = fs::read_to_string("data/shaders/pntriangles.tese").unwrap();
                let file = glsl_to_spirv::compile(&glsl, glsl_to_spirv::ShaderType::TessellationEvaluation)
                    .unwrap();
                let spirv: Vec<u32> = hal::read_spirv(file).unwrap();
                Some(device_state.device.create_shader_module(&spirv).unwrap())
            }
            else {
                None
            };

            let fs_module = {
                let glsl = fs::read_to_string("data/shaders/quad.frag").unwrap();
                let file = glsl_to_spirv::compile(&glsl, glsl_to_spirv::ShaderType::Fragment)
                    .unwrap();
                let spirv: Vec<u32> = hal::read_spirv(file).unwrap();
                device_state.device.create_shader_module(&spirv).unwrap()
            };

            let pipeline = {
                let (vs_entry, fs_entry) = (
                    hal::pso::EntryPoint::<B> {
                        entry: "main",
                        module: &vs_module,
                        specialization: hal::pso::Specialization::default(),
                    },
                    hal::pso::EntryPoint::<B> {
                        entry: "main",
                        module: &fs_module,
                        specialization: hal::pso::Specialization::default(),
                    },
                );

                let tesc_entry = match &tesc_module {
                    Some(tesc_module) => Some(hal::pso::EntryPoint::<B> {
                            entry: "main",
                            module: &tesc_module,
                            specialization: hal::pso::Specialization::default(),
                        }),
                    None => None,
                };

                let tese_entry = match &tese_module {
                    Some(tese_module) => Some(hal::pso::EntryPoint::<B> {
                            entry: "main",
                            module: &tese_module,
                            specialization: hal::pso::Specialization::default(),
                        }),
                    None => None,
                };

                let shader_entries = hal::pso::GraphicsShaderSet {
                    vertex: vs_entry,
                    hull: tesc_entry,
                    domain: tese_entry,
                    geometry: None,
                    fragment: Some(fs_entry),
                };

                let subpass = hal::pass::Subpass {
                    index: 0,
                    main_pass: renderpass,
                };

                let mut pipeline_desc = hal::pso::GraphicsPipelineDesc::new(
                    shader_entries,
                    if use_tessellation { hal::Primitive::PatchList(3) } else { hal::Primitive::TriangleList },
                    hal::pso::Rasterizer {
                        polygon_mode: if use_wireframe { hal::pso::PolygonMode::Line(hal::pso::State::Static(1.0)) } else { hal::pso::PolygonMode::Fill },
                        cull_face: hal::pso::Face::NONE,
                        front_face: hal::pso::FrontFace::CounterClockwise,
                        depth_clamping: false,
                        depth_bias: None,
                        conservative: false,
                    },
                    &new_pipeline_layout,
                    subpass,
                );
                pipeline_desc.blender.targets.push(hal::pso::ColorBlendDesc{
                    mask: hal::pso::ColorMask::ALL,
                    blend: Some(hal::pso::BlendState::ALPHA)
                });
                pipeline_desc.vertex_buffers.push(hal::pso::VertexBufferDesc {
                    binding: 0,
                    stride: std::mem::size_of::<Vertex>() as u32,
                    rate: hal::pso::VertexInputRate::Vertex,
                });

                pipeline_desc.attributes.push(hal::pso::AttributeDesc {
                    location: 0,
                    binding: 0,
                    element: hal::pso::Element {
                        format: hal::format::Format::Rgb32Sfloat,
                        offset: offset_of!(Vertex, a_pos) as u32,
                    },
                });

                pipeline_desc.attributes.push(hal::pso::AttributeDesc {
                    location: 1,
                    binding: 0,
                    element: hal::pso::Element {
                        format: hal::format::Format::Rgba32Sfloat,
                        offset: offset_of!(Vertex, a_col) as u32,
                    },
                });

                pipeline_desc.attributes.push(hal::pso::AttributeDesc {
                    location: 2,
                    binding: 0,
                    element: hal::pso::Element {
                        format: hal::format::Format::Rg32Sfloat,
                        offset: offset_of!(Vertex, a_uv) as u32,
                    },
                });

                pipeline_desc.attributes.push(hal::pso::AttributeDesc {
                    location: 3,
                    binding: 0,
                    element: hal::pso::Element {
                        format: hal::format::Format::Rgb32Sfloat,
                        offset: offset_of!(Vertex, a_norm) as u32,
                    },
                });

                pipeline_desc.attributes.push(hal::pso::AttributeDesc {
                    location: 4,
                    binding: 0,
                    element: hal::pso::Element {
                        format: hal::format::Format::Rgba32Sfloat,
                        offset: offset_of!(Vertex, a_joint_indices) as u32,
                    },
                });

                pipeline_desc.attributes.push(hal::pso::AttributeDesc {
                    location: 5,
                    binding: 0,
                    element: hal::pso::Element {
                        format: hal::format::Format::Rgba32Sfloat,
                        offset: offset_of!(Vertex, a_joint_weights) as u32,
                    },
                });

                pipeline_desc.depth_stencil.depth = Some(hal::pso::DepthTest {
                    fun: hal::pso::Comparison::GreaterEqual,
                    write: true,
                });
                pipeline_desc.depth_stencil.depth_bounds = false;
                pipeline_desc.depth_stencil.stencil = None;

                device_state.device.create_graphics_pipeline(&pipeline_desc, None)
            };

            device_state.device.destroy_shader_module(vs_module);
            device_state.device.destroy_shader_module(fs_module);
            match tesc_module {
                Some(tesc_module) => {device_state.device.destroy_shader_module(tesc_module)},
                None => {},
            }
            match tese_module {
                Some(tese_module) => {device_state.device.destroy_shader_module(tese_module)},
                None => {},
            }
            pipeline.unwrap()
        };

        (new_pipeline, new_pipeline_layout)
    };

    //Pipeline permuations
    let (pipeline, pipeline_layout) = create_pipeline(&device_state, &renderpass, &set_layout, false, false);
    let (wireframe_pipeline, wireframe_pipeline_layout) = create_pipeline(&device_state, &renderpass, &set_layout, true, false);
    let(tess_pipeline, tess_pipeline_layout) = create_pipeline(&device_state, &renderpass, &set_layout, false, true);
    let (tess_wire_pipeline, tess_wire_pipeline_layout) = create_pipeline(&device_state, &renderpass, &set_layout, true, true);

	//initialize cimgui
	let mut cimgui_hal = CimguiHal::new( &device_state, &mut general_queue_group, &format, &depth_format);

    #[derive(Debug, Clone, Copy, Default)]
    #[repr(C)]
    struct Vec4 {
        x : f32,
        y : f32,
        z : f32,
        w : f32,
    }

    #[derive(Debug, Clone, Copy, Default)]
    #[repr(C)]
    struct DCVert {
        position : Vec4,
        normal   : Vec4,
    }

    let voxel_size = 16.0;
    let voxel_dimensions : [u32;3] = [10,10,10];
    let total_voxels = voxel_dimensions.iter().product::<u32>() as usize;

    let chunk_dimensions : [i32;3] = [10, 6, 10];
    let _total_chunks = chunk_dimensions.iter().map(|x| x * 2).product::<i32>() as usize;

    //TODO: glsl->spirv->shader module helper function in gfx_helpers
    let shader_module = {
        let compute_shader_spirv : Vec<u32> = {
            let glsl = fs::read_to_string("data/shaders/generate_vertices.comp").expect("failed to read compute shader code to string");
            let file = glsl_to_spirv::compile(&glsl, glsl_to_spirv::ShaderType::Compute)
                    .unwrap();
            hal::read_spirv(file).unwrap()
        };
        #[allow(unused_unsafe)]
        unsafe {
            device_state.device.create_shader_module(&compute_shader_spirv).expect("failed to create compute shader module")
        }
    };

    let mut pool = Pool::new((num_cpus::get() - 1) as u32);

    let mut dc_meshes = Vec::new();
    let (tx, rx) = channel();

    pool.scoped(|scoped| {

    let device_state =  Arc::new(&device_state);
    let compute_queue_group = Arc::new(Mutex::new(&mut compute_queue_group));

    //FIXME: these build async to rendering now, but will stall quitting when trying to close the window
    for y in -chunk_dimensions[1]..chunk_dimensions[1] {
        for x in -chunk_dimensions[0]..chunk_dimensions[0] {
            for z in -chunk_dimensions[2]..chunk_dimensions[2] {
    
                let device_state = device_state.clone();
                let compute_queue_group = compute_queue_group.clone();
                let tx = tx.clone();
                let shader_module = Arc::new(&shader_module);

                scoped.execute(move || {

                    let vertices_buffer = GpuBuffer::new_cpu_visible(
                        &vec![DCVert::default(); total_voxels], 
                        hal::buffer::Usage::STORAGE, 
                        &device_state
                    );

                    let indices_buffer = GpuBuffer::new_cpu_visible(
                        &vec![-1i32; total_voxels * 18 ],
                        hal::buffer::Usage::STORAGE, 
                        &device_state
                    );

                    let compute_uniform_buffer = GpuBuffer::new_cpu_visible(
                        &vec![
                            //Voxel Offset
                            (Vec4 { 
                                x: (x * (voxel_dimensions[0] - 1) as i32) as f32 * voxel_size,
                                y: (y * (voxel_dimensions[1] - 1) as i32) as f32 * voxel_size,
                                z: (z * (voxel_dimensions[2] - 1) as i32) as f32 * voxel_size,
                                w: 1.0,
                            },
                            [voxel_size, 0.0, 0.0, 0.0],
                            voxel_dimensions)
                        ],
                        hal::buffer::Usage::UNIFORM, 
                        &device_state
                    );

                    let compute_context = ComputeContext::new(
                        &shader_module,
                        voxel_dimensions,
                        vec![vertices_buffer, indices_buffer, compute_uniform_buffer],
                        &device_state, 
                        &compute_queue_group.lock().unwrap()
                    );

                    compute_context.dispatch(&mut compute_queue_group.lock().unwrap().queues[0]);
                    compute_context.wait_for_completion(&device_state);

                    //FIXME: currently converting this data to gltf_model vertex data for quick testing
                    let vertex_data : Vec<Vertex> = compute_context.buffers[0].get_data::<DCVert>(&device_state).iter().map(|v| {     
                        Vertex {
                        a_pos : [v.position.x, v.position.y, v.position.z],
                        a_col: [(x.abs() % 2) as f32 + 0.5, (y.abs() % 2) as f32 + 0.5, (z.abs() % 2) as f32 + 0.5, 1.0],
                        a_uv:  [0.0, 0.0],
                        a_norm: [v.normal.x, v.normal.y, v.normal.z],
                        a_joint_indices: [0.0, 0.0, 0.0, 0.0],
                        a_joint_weights: [0.0, 0.0, 0.0, 0.0],
                        }
                    }).collect();

                    //TODO: faster way to filter data
                    let index_data : Vec<u32> = compute_context.buffers[1].get_data::<i32>(&device_state).iter().filter(|&&i| i != -1).map(|i| *i as u32).collect();
                    if !index_data.is_empty() {
                        tx.send(
                            (GpuBuffer::new_cpu_visible(&vertex_data, hal::buffer::Usage::VERTEX, &device_state),
                            GpuBuffer::new_cpu_visible(&index_data, hal::buffer::Usage::INDEX, &device_state))
                        ).expect("failed to send mesh data across channel");
                    }

                    compute_context.destroy(&device_state, true);
                });
            }
        }
    }

    let acquisition_semaphore = device_state.device.create_semaphore().unwrap();

    let mut frame_fence = device_state.device.create_fence(false).unwrap();

    let mut running = true;
    let mut needs_resize = false;
    let (mut window_width, mut window_height) = (0u32, 0u32);

    let first_timestamp = timestamp();
    let mut last_time = 0.0f64;

    //Key Hashmap
    let mut key_states = HashMap::new();

    let mut num_frames = 0;

    let mut is_fullscreen = false;

	let mut mouse_pos = [0.0, 0.0];
	let mut mouse_button_states =  [ false, false, false, false, false];
	let mut anim_speed : f32 = 1.0;
    let mut draw_wireframe = false;
    let mut use_tessellation = false;
    let mut max_terrain_triangles : f32 = 100000000.0;

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
                                key_states.insert(keycode,input.state == winit::ElementState::Pressed);
								match keycode {
									winit::VirtualKeyCode::F11 => {
										if input.state == winit::ElementState::Pressed {
											is_fullscreen = !is_fullscreen;
											if is_fullscreen {
												window.set_fullscreen(Some(window.get_primary_monitor()));
											}
											else {
												window.set_fullscreen(None);
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

        if let Some(true) = key_states.get(&winit::VirtualKeyCode::W) {
            forward += 1.0;
        }
        if let Some(true) = key_states.get(&winit::VirtualKeyCode::S) {
            forward -= 1.0;
        }
        if let Some(true) = key_states.get(&winit::VirtualKeyCode::D) {
            right += 1.0;
        }
        if let Some(true) = key_states.get(&winit::VirtualKeyCode::A) {
            right -= 1.0;
        }
        if let Some(true) = key_states.get(&winit::VirtualKeyCode::E) {
            up += 1.0;
        }
        if let Some(true) = key_states.get(&winit::VirtualKeyCode::Q) {
            up -= 1.0;
        }
        if let Some(true) = key_states.get(&winit::VirtualKeyCode::LShift) {
            forward *= 50.0;
            right *= 50.0;
            up *= 50.0;
        }

        if window_width == 0 || window_height == 0 {
            continue;
        }
        
        let needed_resize = needs_resize;
        if needs_resize {
            device_state.device.wait_idle().unwrap();

            //Destroy old resources
            for framebuffer in framebuffers {
                device_state.device.destroy_framebuffer(framebuffer);
            }

            for (_, rtv) in frame_images {
                device_state.device.destroy_image_view(rtv);
            }

            device_state.device.destroy_image_view(depth_view);
            device_state.device.destroy_image(depth_image);
            device_state.device.free_memory(depth_memory);

            device_state.device.destroy_swapchain(swapchain);

            //Build new resources         
            let (new_swapchain, new_backbuffer, new_format, new_extent) = create_swapchain(&device_state, &mut surface );

            swapchain = new_swapchain;
            format = new_format;
            extent = new_extent;

            let (new_depth_view, new_depth_image, new_depth_memory, new_depth_format) = create_depth_buffer(&device_state, &extent, false);
            depth_view = new_depth_view;
            depth_image = new_depth_image;
            depth_memory = new_depth_memory;
            depth_format = new_depth_format;

            let (new_frame_images, new_framebuffers) = create_framebuffers(&device_state, new_backbuffer, &format, &extent, &depth_view, &renderpass);
            frame_images = new_frame_images;
            framebuffers = new_framebuffers;

            //Update Projection matrix (possible change in aspect ratio)
            general_uniform_struct.proj_matrix = glm::perspective_zo(
                extent.width as f32 / extent.height as f32,
                degrees_to_radians(90.0f32),
                100000.0,
                0.01
            ).into();
            
            //flipping this to make y point up
            general_uniform_struct.proj_matrix[1][1] *= -1.0;

            needs_resize = false;
            //TODO: figure out why this fixes our Renderpass RenderArea/Framebuffer size discrepancy issue
            continue;
        }

        let time = timestamp() - first_timestamp;
        let delta_time = time - last_time;

        //Rotate cam_forward & cam_up when right mouse pressed using mouse delta
        if mouse_button_states[0] && !cimgui_hal.wants_capture_mouse() {
            let yaw_rotation = glm::quat_angle_axis(degrees_to_radians(-mouse_delta.0 * 100.0 * delta_time) as f32, &glm::vec3(0.,1.,0.));
            let pitch_rotation = glm::quat_angle_axis(degrees_to_radians(-mouse_delta.1 * 100.0 * delta_time) as f32, &cam_forward.cross(&cam_up));

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

        general_uniform_struct.view_matrix = glm::look_at(
            &cam_pos,
            &(cam_pos + cam_forward),
            &cam_up
        ).into();

        general_uniform_struct.time = time as f32;

		uniform_gpu_buffer.reupload(&[general_uniform_struct], &device_state, &mut general_queue_group);

        //Animate Bones
        gltf_model.animate(0, 0, delta_time *  anim_speed as f64);

		//Upload Bones to GPU
		gltf_model.upload_bones(&device_state, &mut general_queue_group);
        
        device_state.device.reset_fence(&frame_fence).unwrap();
        command_pool.reset(false);

        let frame: hal::SwapImageIndex = {
            match swapchain.acquire_image(!0, Some(&acquisition_semaphore), None) {
                Ok((i,_)) => i,
                Err(_) => {
                    needs_resize = true;
                    continue;
                }
            }
        };

		let mut cmd_buffer = command_pool.acquire_command_buffer::<hal::command::OneShot>();
		cmd_buffer.begin();

		let viewport = hal::pso::Viewport {
			rect: hal::pso::Rect {
				x: 0,
				y: 0,
				w: extent.width as i16,
				h: extent.height as i16,
			},
			depth: 0.0..1.0,
		};

        if needed_resize {
            println!("Viewport Dimensions {} {}", viewport.rect.w, viewport.rect.h);
        }

        //Collect DC_Meshes that we completed since last frame
        while let Ok((dc_vertex_buffer, dc_index_buffer)) = rx.try_recv() {
            dc_meshes.push((dc_vertex_buffer, dc_index_buffer));
        }

        let mut dc_mesh_indices_rendered = 0;

        //Shadow Pass
        {
            let shadow_viewport = hal::pso::Viewport {
                rect: hal::pso::Rect {
                    x: 0,
                    y: 0,
                    w: shadow_map_extent.width as i16,
                    h: shadow_map_extent.height as i16,
                },
                depth: 0.0..1.0,
            };

            cmd_buffer.set_viewports(0, &[shadow_viewport.clone()]);
            cmd_buffer.set_scissors(0, &[shadow_viewport.rect]);
            cmd_buffer.bind_graphics_pipeline(&shadow_pipeline);
            cmd_buffer.bind_graphics_descriptor_sets(&shadow_pipeline_layout, 0, Some(&shadow_desc_set), &[]);

            let mut shadow_map_encoder = cmd_buffer.begin_render_pass_inline(
                &shadow_renderpass,
                &shadow_framebuffer,
                shadow_viewport.rect,
                &[
                    hal::command::ClearValue::DepthStencil(hal::command::ClearDepthStencil(0.0, 0))
                ],
            );

            gltf_model.record_draw_commands(&mut shadow_map_encoder, 100);

            //Dual Contour Testing
            for (dc_vertex_buffer, dc_index_buffer) in &dc_meshes {
                
                //Stop Rendering early if we've reached the max value of terrain we'd like to render
                if dc_mesh_indices_rendered / 3 > max_terrain_triangles as u32 {
                    break;
                }
                
                shadow_map_encoder.bind_vertex_buffers(0, Some((&dc_vertex_buffer.buffer, 0)));
				shadow_map_encoder.bind_index_buffer(hal::buffer::IndexBufferView {
                    buffer: &dc_index_buffer.buffer,
                    offset: 0,
                    index_type: hal::IndexType::U32,
                });
                shadow_map_encoder.draw_indexed(0..dc_index_buffer.count, 0, 0..1);
                dc_mesh_indices_rendered += dc_index_buffer.count;
            }
        }

        //Main Rendering Pass
		{
            cmd_buffer.set_viewports(0, &[viewport.clone()]);
            cmd_buffer.set_scissors(0, &[viewport.rect]);

            //Bind correct pipeline and use correct layout
            match (draw_wireframe, use_tessellation) {
                (true, false) => {
                    cmd_buffer.bind_graphics_pipeline(&wireframe_pipeline);
                    cmd_buffer.bind_graphics_descriptor_sets(&wireframe_pipeline_layout, 0, Some(&desc_set), &[]);
                },
                (false, false) => {
                    cmd_buffer.bind_graphics_pipeline(&pipeline);
                    cmd_buffer.bind_graphics_descriptor_sets(&pipeline_layout, 0, Some(&desc_set), &[]);
                },
                (false, true) => {
                    cmd_buffer.bind_graphics_pipeline(&tess_pipeline);
                    cmd_buffer.bind_graphics_descriptor_sets(&tess_pipeline_layout, 0, Some(&desc_set), &[]);
                },
                (true, true) => {
                    cmd_buffer.bind_graphics_pipeline(&tess_wire_pipeline);
                    cmd_buffer.bind_graphics_descriptor_sets(&tess_wire_pipeline_layout, 0, Some(&desc_set), &[]);
                }
            }

			let mut encoder = cmd_buffer.begin_render_pass_inline(
				&renderpass,
				&framebuffers[frame as usize],
				viewport.rect,
				&[
					hal::command::ClearValue::Color(hal::command::ClearColor::Sfloat([0.2, 0.2, 0.2, 0.0,])),
					hal::command::ClearValue::DepthStencil(hal::command::ClearDepthStencil(0.0, 0))
				],
			);

			gltf_model.record_draw_commands(&mut encoder, 100);

            //Dual Contour Testing
            dc_mesh_indices_rendered = 0;
            for (dc_vertex_buffer, dc_index_buffer) in &dc_meshes {
                
                //Stop Rendering early if we've reached the max value of terrain we'd like to render
                if dc_mesh_indices_rendered / 3 > max_terrain_triangles as u32 {
                    break;
                }
                
                encoder.bind_vertex_buffers(0, Some((&dc_vertex_buffer.buffer, 0)));
				encoder.bind_index_buffer(hal::buffer::IndexBufferView {
                    buffer: &dc_index_buffer.buffer,
                    offset: 0,
                    index_type: hal::IndexType::U32,
                });
                encoder.draw_indexed(0..dc_index_buffer.count, 0, 0..1);
                dc_mesh_indices_rendered += dc_index_buffer.count;
            }
		}

		cimgui_hal.new_frame(window_width as f32, window_height as f32, delta_time as f32);

		//TODO: Safe API for cimgui
		#[allow(unused_unsafe)]
		unsafe {
			use std::ffi::CString;

			igBegin(CString::new("GFX-RS TESTING").unwrap().as_ptr(), &mut true, 0);
			igText(CString::new("Drag the Slider to change anim speed").unwrap().as_ptr());
			igSliderFloat(CString::new("Anim Speed").unwrap().as_ptr(), &mut anim_speed, 0.0f32, 15.0f32, std::ptr::null(), 2.0f32);
            igText(CString::new(format!("Terrain Triangle Count: {}", dc_mesh_indices_rendered / 3)).unwrap().as_ptr());
            igSliderFloat(CString::new("Terrain Triangle Cutoff").unwrap().as_ptr(), &mut max_terrain_triangles, 0.0, 100000000.0, std::ptr::null(), 6.0f32);
			igSliderFloat(CString::new("PN Triangles Strength").unwrap().as_ptr(), &mut general_uniform_struct.pn_triangles_strength, 0.0, 1.0, std::ptr::null(), 1.0f32);
            igCheckbox(CString::new("Use Tessellation").unwrap().as_ptr(), &mut use_tessellation);
            if use_tessellation {
                if  general_uniform_struct.tess_level < 0.0 { general_uniform_struct.tess_level = 1.0; }
                igSliderFloat(CString::new("Tess Level").unwrap().as_ptr(), &mut general_uniform_struct.tess_level, 1.0, 8.0, std::ptr::null(), 1.0f32);
            }
            else {
                general_uniform_struct.tess_level = -1.0;
            }
            igCheckbox(CString::new("Wireframe").unwrap().as_ptr(), &mut draw_wireframe);
            igSliderFloat(CString::new("Shadow Bias").unwrap().as_ptr(), &mut shadow_uniform_struct.bias, 0.0, 0.015, std::ptr::null(), 1.0f32);
            igEnd();

            shadow_uniform_buffer.reupload(&[shadow_uniform_struct], &device_state, &mut general_queue_group);

			igShowDemoWindow(&mut true);
		}

        //Cimgui Pass
		cimgui_hal.render(&mut cmd_buffer, &framebuffers[frame as usize], &device_state, &mut general_queue_group);

		cmd_buffer.finish();

        //TODO: move out of render loop
        let submission_semaphore = device_state.device.create_semaphore().unwrap();

        {
			let submission = hal::queue::Submission {
                command_buffers: Some(&cmd_buffer),
                wait_semaphores: Some((&acquisition_semaphore, hal::pso::PipelineStage::BOTTOM_OF_PIPE)),
                signal_semaphores: Some(&submission_semaphore),
            };
            general_queue_group.queues[0].submit(submission, Some(&mut frame_fence));
        }

        //TODO: Remove this and fix synchro bugs(last time i tried there were some weird synchro issues when going full screen on DX12 backend)
        //FIXME: if we remove this, need to use 1 command_buffer per frame
        device_state.device.wait_for_fence(&frame_fence, !0).unwrap();

        // present frame
        if let Err(_) = swapchain.present(&mut general_queue_group.queues[0], frame, &[submission_semaphore]) {
            needs_resize = true;
        }

        last_time = time;
    }

    let total_time = timestamp() - first_timestamp;
    println!("Avg Frame Time: {}", total_time / num_frames as f64);

    device_state.device.wait_idle().unwrap();

    device_state.device.destroy_semaphore(acquisition_semaphore);

    gltf_model.destroy(&device_state);

	cimgui_hal.destroy(&device_state);

    device_state.device.destroy_command_pool(command_pool.into_raw());

    device_state.device.destroy_descriptor_pool(desc_pool);
    device_state.device.destroy_descriptor_set_layout(set_layout);
    device_state.device.destroy_render_pass(renderpass);
    device_state.device.destroy_graphics_pipeline(pipeline);
    device_state.device.destroy_pipeline_layout(pipeline_layout);

    //Destroy old resources
    for framebuffer in framebuffers {
        device_state.device.destroy_framebuffer(framebuffer);
    }

    for (_, rtv) in frame_images {
        device_state.device.destroy_image_view(rtv);
    }

    device_state.device.destroy_swapchain(swapchain);

    });

    //Compute Shader Module
    device_state.device.destroy_shader_module(shader_module);

    for (vertex_buffer, index_buffer) in dc_meshes {
        vertex_buffer.destroy(&device_state);
        index_buffer.destroy(&device_state);
    }
    
	}
}

#[cfg_attr(rustfmt, rustfmt_skip)]
const DIMS: hal::window::Extent2D = hal::window::Extent2D { width: 1280, height: 720 };

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct UniformStruct {
    view_matrix:  [[f32;4];4],
    proj_matrix:  [[f32;4];4],
    model_matrix: [[f32;4];4],
    time: f32,
    pn_triangles_strength : f32,
    tess_level : f32,
}

fn timestamp() -> f64 {
    let timespec = time::get_time();
    timespec.sec as f64 + (timespec.nsec as f64 / 1000.0 / 1000.0 / 1000.0)
}

fn degrees_to_radians<T>( deg: T) -> T 
where T: num::Float {
    deg * num::cast(0.0174533).unwrap()
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct ShadowUniform {
    shadow_mvp:  [[f32;4];4],
    light_dir : [f32;3],
    bias : f32,
}