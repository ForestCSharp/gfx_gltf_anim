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
extern crate gfx_hal as hal;

extern crate num;
extern crate glsl_to_spirv;
extern crate winit;
extern crate gltf;
extern crate nalgebra_glm as glm;
extern crate rand;
extern crate time;

#[macro_use]
extern crate memoffset;

use std::fs;
use std::io::{Read};
use std::collections::HashMap;

use hal::{Instance, PhysicalDevice, Device, DescriptorPool, Surface, Swapchain};

#[cfg(any(feature = "vulkan", feature = "dx12", feature = "metal"))]
fn main() {

    println!("Current Target: {}", env!("TARGET"));

    //Load GLTF Model
    let (gltf_model, buffers, _) = gltf::import("data/CesiumMan.gltf").unwrap();

    //TODO: Handle multiple meshes/primitives (need to wrap VBuf/Ibuf/Option<Skeleton> into struct)
    let mut vertices_vec = Vec::new();
    let mut indices_vec = Vec::new();
    let mut skeleton : GpuSkeleton = GpuSkeleton::new();

    //TODO: store each animation (Only getting first anim for now)
    let mut animations : Vec<gltf::Animation> = gltf_model.animations().collect();
    let anim = animations.remove(0);

    //TODO: move to struct: An animation is simply a bunch of translation, rotation, and scale channels
    let mut anim_channels = Vec::new();
    let mut anim_duration = 0.0;

    //Store the animation
    for channel in anim.channels() {       
        //Channel Reader
        let channel_reader = channel.reader(|buffer| Some(&buffers[buffer.index()]));

        let node_index = channel.target().node().index();

        let times = channel_reader.read_inputs().unwrap();

        match channel_reader.read_outputs().unwrap() {
            gltf::animation::util::ReadOutputs::Translations(mut translations) => {
                
                let mut translation_keyframes = Vec::new();
                
                for (time, translation) in times.zip(translations) {
                    translation_keyframes.push((time, translation));
                }

                anim_channels.push((node_index, AnimChannel {
                    keyframes : ChannelType::TranslationChannel(translation_keyframes),
                    current_left_keyframe: 0,
                }));
            },
            gltf::animation::util::ReadOutputs::Rotations(mut rotations) => {
                
                let mut rotation_keyframes = Vec::new();

                for (time, rotation) in times.zip(rotations.into_f32()) {
                    rotation_keyframes.push((time, rotation));
                }

                anim_channels.push((node_index, AnimChannel {
                    keyframes: ChannelType::RotationChannel(rotation_keyframes),
                    current_left_keyframe: 0,
                }));
            },
            gltf::animation::util::ReadOutputs::Scales(mut scales) => {
                
                let mut scale_keyframes = Vec::new();
                
                for (time, scale) in times.zip(scales) {
                    scale_keyframes.push((time, scale));
                }

                anim_channels.push((node_index, AnimChannel {
                    keyframes: ChannelType::ScaleChannel(scale_keyframes),
                    current_left_keyframe: 0,
                }));
            },
            _ => {
                println!("Unsupported Anim Channel");
            },
        }

        //Get Anim Duration
        for time_val in channel_reader.read_inputs().unwrap() {
           if time_val > anim_duration { anim_duration = time_val; }
        }
    }

    //Store all nodes (their index == index in vec, parent index, children indices, and transform)
    let mut nodes = Vec::new();

    //Map child indices to parent indices (used below when building up node Vec)
    let node_parents = get_node_parents(&mut gltf_model.nodes());

    for node in gltf_model.nodes() {

        let children_indices = node.children().map(|child| child.index()).collect::<Vec<usize>>();

        let (translation, rotation, scale) = node.transform().decomposed();

        nodes.push(
            Node {
                parent: node_parents[&node.index()],
                children: children_indices,
                translation: translation,
                rotation: rotation,
                scale: scale,
            }
        );

        let parent_index = match nodes[nodes.len()-1].parent {
            Some(index) => index.to_string(),
            None => "N/A".to_string(),
        };

        println!("INDEX: {},\tPARENT: {},\tCHILDREN {:?}", node.index(), parent_index , nodes[nodes.len() - 1].children);
    }

    for node in gltf_model.nodes() {

        let has_mesh = node.mesh().is_some();
        let is_skinned = node.skin().is_some();

        //When a node contains a skin, all its meshes contain JOINTS_0 and WEIGHTS_0 attributes.

        match node.mesh() {
            Some(mesh) => {
                for primitive in mesh.primitives() {

                    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
                    let pos_iter = reader.read_positions().unwrap(); 
                    //TODO: Better error handling if no positions

                    //Optional Colors
                    let mut col_iter = match reader.read_colors(0) {
                        Some(col_iter) => Some(col_iter.into_rgba_f32()),
                        None => None,
                    };

                    //Optional UVs (TODO: Warn if mesh doesn't have these)
                    let mut uv_iter = match reader.read_tex_coords(0) {
                        Some(uv_iter) => Some(uv_iter.into_f32()),
                        None => None,
                    };

                    //Normals
                    let mut norm_iter = reader.read_normals();

                    //if skinned, we need to get the JOINTS_0 and WEIGHTS_0 attributes
                    let mut joints_iter = match reader.read_joints(0) {
                        Some(joints_iter) => Some(joints_iter.into_u16()),
                        None => None,
                    };

                    let mut weights_iter = match reader.read_weights(0) {
                        Some(weights_iter) => Some(weights_iter.into_f32()),
                        None => None,
                    };

                    //Iterate over our positions
                    for pos in pos_iter {

                        let col = match &mut col_iter {
                            Some(col_iter) =>  match col_iter.next() {
                                Some(col) => col,
                                None => [0., 0., 0., 1.0],
                            },
                            None => [0., 0., 0., 1.0],
                        };

                        let uv = match &mut uv_iter {
                            Some(uv_iter) => match uv_iter.next() {
                                Some(uv) => uv,
                                None => [0.0, 0.0],
                            },
                            None => [0.0, 0.0],
                        };

                        let norm = match &mut norm_iter {
                            Some(norm_iter) => match norm_iter.next() {
                                Some(norm) => norm,
                                None => [0.0, 0.0, 0.0],
                            },
                            None => [0.0, 0.0, 0.0],
                        };

                        let joint_indices = match &mut joints_iter {
                            Some(joints_iter) => match joints_iter.next() {
                                Some(joint_indices) => [
                                            joint_indices[0] as f32, 
                                            joint_indices[1] as f32, 
                                            joint_indices[2] as f32, 
                                            joint_indices[3] as f32,
                                            ],
                                None => [0., 0., 0., 0.],
                            },
                            None => [0., 0., 0., 0.],
                        };

                        let joint_weights = match &mut weights_iter {
                            Some(weights_iter) => match weights_iter.next() {
                                Some(joint_weights) => joint_weights,
                                None => [0.0, 0.0, 0.0, 0.0],
                            },
                            None => [0.0, 0.0, 0.0, 0.0],
                        };

                        //println!("Joint Indices: {:?}", joint_indices);
                        //println!("Joint Weights: {:?} \n", joint_weights);

                        vertices_vec.push( Vertex { 
                            a_pos: pos,
                            a_col: col,
                            a_uv: uv,
                            a_norm: norm,
                            a_joint_indices: joint_indices,
                            a_joint_weights: joint_weights,
                        });
                    }

                    //Indices
                    indices_vec = reader.read_indices().map( |read_indices| {
                        read_indices.into_u32().collect()
                    }).unwrap();
                    //TODO: Better handling of this (not all GLTF meshes have indices)
                } 
            },
            None => {},
        }

        //skinning: build up skeleton
        if has_mesh && is_skinned {
            match node.skin() {
                Some(skin) => {
                    let reader = skin.reader(|buffer| Some(&buffers[buffer.index()]));
                    //If "None", then each joint's inv_bind_matrix is assumed to be 4x4 Identity matrix
                    let mut inverse_bind_matrices = reader.read_inverse_bind_matrices();

                    skeleton.inverse_root_transform = glm::inverse(&node.transform().matrix().into());
                    
                    //Joints are nodes
                    for joint in skin.joints() {

                        let inverse_bind_matrix: glm::Mat4 = {

                            let mut out_matrix : glm::Mat4 = glm::Mat4::identity();

                            match &mut inverse_bind_matrices {
                                Some(inverse_bind_matrices) => {
                                    match inverse_bind_matrices.next() {
                                        Some(matrix) => out_matrix = matrix.into(),
                                        None => {}, //used up our iterator
                                    }
                                },
                                None => {}, //iterator was none (assume 4x4 identity matrix)
                            } 

                            out_matrix
                        };
                        
                        //Build up skeleton
                        let joint_transform : glm::Mat4 = compute_global_transform(joint.index(), &nodes);

                        let joint_matrix = skeleton.inverse_root_transform * joint_transform * inverse_bind_matrix;

                        skeleton.bones.push(GpuBone {
                            joint_matrix: joint_matrix.into(),
                        });

                        skeleton.inverse_bind_matrices.push(inverse_bind_matrix);

                        //map index
                        skeleton.gpu_index_to_node_index.insert(skeleton.bones.len() - 1, joint.index());
                    }
                },
                None => {},
            }
        }
    }

    //Create a window builder
    let window_builder = winit::WindowBuilder::new()
        .with_dimensions(winit::dpi::LogicalSize::new(INITIAL_WIDTH, INITIAL_HEIGHT))
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
    let mut adapter = adapters.remove(0);
    let memory_types = adapter.physical_device.memory_properties().memory_types;
    let _limits = adapter.physical_device.limits();

    //Create Device and Queue from our adapter
    let (mut device, mut queue_group) = adapter
        .open_with::<_, hal::Graphics>(1, |family| surface.supports_queue_family(family))
        .unwrap();

    let mut command_pool = device.create_command_pool_typed(&queue_group, hal::pool::CommandPoolCreateFlags::empty(), 16);

    //Descriptor Set
    let set_layout = device.create_descriptor_set_layout( 
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
    );
    
    let mut desc_pool = device.create_descriptor_pool(
        1,
        &[
            hal::pso::DescriptorRangeDesc {
                    ty: hal::pso::DescriptorType::UniformBuffer,
                    count: 2,
                },
        ],
    );

    let desc_set = desc_pool.allocate_set(&set_layout).unwrap();

    #[derive(Debug, Clone, Copy)]
    struct CameraUniform {
        view_matrix:  [[f32;4];4],
        proj_matrix:  [[f32;4];4],
        model_matrix: [[f32;4];4],
        time: f32,
    }

    let mut cam_pos = glm::vec3(1.0, 0.0, -0.5);
    let mut cam_forward = glm::vec3(0.,0.,0.,) - cam_pos;
    let mut cam_up = glm::vec3(0., 1., 0.);

    let view_matrix = glm::look_at_rh(
        &cam_pos,
        &(cam_pos + cam_forward),
        &cam_up
    );

    let perspective_matrix = glm::perspective(
        INITIAL_WIDTH as f32 / INITIAL_HEIGHT as f32,
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
    let uniform_buffer_len = std::mem::size_of::<CameraUniform>() as u64;
    let uniform_buffer_unbound = device.create_buffer(uniform_buffer_len, hal::buffer::Usage::UNIFORM).unwrap();
    let uniform_buffer_req = device.get_buffer_requirements(&uniform_buffer_unbound);

    let uniform_upload_type = memory_types
        .iter()
        .enumerate()
        .position(|(id, mem_type)| {
            uniform_buffer_req.type_mask & (1 << id) != 0
                && mem_type.properties.contains(hal::memory::Properties::CPU_VISIBLE)
        }).unwrap().into();

    let uniform_buffer_memory = device.allocate_memory(uniform_upload_type, uniform_buffer_req.size).unwrap();
    let uniform_buffer = device.bind_buffer_memory(&uniform_buffer_memory, 0, uniform_buffer_unbound).unwrap();

    {
        let mut uniform_writer = device.acquire_mapping_writer::<CameraUniform>(&uniform_buffer_memory, 0..uniform_buffer_req.size).unwrap();
        uniform_writer[0] = camera_uniform_struct; 
        device.release_mapping_writer(uniform_writer);
    }

    //Skeleton Uniform Buffer Setup
    //TODO: Don't try to create this buffer if no bones (len() == 0)
    //FIXME: Causes crash if no bones (i.e. unskinned models)
    let skeleton_uniform_len = (std::cmp::max(1,skeleton.bones.len()) * std::mem::size_of::<GpuBone>()) as u64;
    let skeleton_uniform_unbound = device.create_buffer(skeleton_uniform_len, hal::buffer::Usage::UNIFORM).unwrap();
    let skeleton_uniform_req = device.get_buffer_requirements(&skeleton_uniform_unbound);

    let skeleton_upload_type = memory_types
        .iter()
        .enumerate()
        .position(|(id, mem_type)| {
            skeleton_uniform_req.type_mask & (1 << id) != 0
                && mem_type.properties.contains(hal::memory::Properties::CPU_VISIBLE)
        }).unwrap().into();

    let skeleton_uniform_memory = device.allocate_memory(skeleton_upload_type, skeleton_uniform_req.size).unwrap();
    let skeleton_uniform_buffer = device.bind_buffer_memory(&skeleton_uniform_memory, 0, skeleton_uniform_unbound).unwrap();

    {
        let mut uniform_writer = device.acquire_mapping_writer::<GpuBone>(&skeleton_uniform_memory, 0..skeleton_uniform_req.size).unwrap();
        uniform_writer[0..skeleton.bones.len()].copy_from_slice(&skeleton.bones);
        device.release_mapping_writer(uniform_writer);
    }

    //Descriptor write for our two uniform buffers
    device.write_descriptor_sets( vec![
        hal::pso::DescriptorSetWrite {
            set: &desc_set,
            binding: 0,
            array_offset: 0,
            descriptors: Some(hal::pso::Descriptor::Buffer(&uniform_buffer, None..None)),
        },
        hal::pso::DescriptorSetWrite {
            set: &desc_set,
            binding: 1,
            array_offset: 0,
            descriptors: Some(hal::pso::Descriptor::Buffer(&skeleton_uniform_buffer, None..None)),
        },
    ]);

    //Vertex Buffer Setup
    let buffer_stride = std::mem::size_of::<Vertex>() as u64;
    let buffer_len = vertices_vec.len() as u64 * buffer_stride;
    let buffer_unbound = device.create_buffer(buffer_len, hal::buffer::Usage::VERTEX).unwrap();
    let buffer_req = device.get_buffer_requirements(&buffer_unbound);

    let upload_type = memory_types
        .iter()
        .enumerate()
        .position(|(id, mem_type)| {
            buffer_req.type_mask & (1 << id) != 0
                && mem_type.properties.contains(hal::memory::Properties::CPU_VISIBLE)
        }).unwrap().into();

    let buffer_memory = device.allocate_memory(upload_type, buffer_req.size).unwrap();
    let vertex_buffer = device.bind_buffer_memory(&buffer_memory, 0, buffer_unbound).unwrap();

    {
        let mut vertices = device.acquire_mapping_writer::<Vertex>(&buffer_memory, 0..buffer_req.size).unwrap();
        vertices[0..vertices_vec.len()].copy_from_slice(&vertices_vec);
        device.release_mapping_writer(vertices);
    }

    //Index Buffer Setup
    let index_buffer_stride = std::mem::size_of::<u32>() as u64;
    let index_buffer_len = indices_vec.len() as u64 * index_buffer_stride;
    let index_buffer_unbound = device.create_buffer(index_buffer_len, hal::buffer::Usage::INDEX).unwrap();
    let index_buffer_req = device.get_buffer_requirements(&index_buffer_unbound);

    let index_upload_type = memory_types
        .iter()
        .enumerate()
        .position(|(id, mem_type)| {
            index_buffer_req.type_mask & (1 << id) != 0
                && mem_type.properties.contains(hal::memory::Properties::CPU_VISIBLE)
        }).unwrap().into();

    let index_buffer_memory = device.allocate_memory(index_upload_type, index_buffer_req.size).unwrap();
    let index_buffer = device.bind_buffer_memory(&index_buffer_memory, 0, index_buffer_unbound).unwrap();
    {
        let mut indices = device.acquire_mapping_writer::<u32>(&index_buffer_memory, 0..index_buffer_req.size).unwrap();
        indices[0..indices_vec.len()].copy_from_slice(&indices_vec);
        device.release_mapping_writer(indices);
    }

    let create_swapchain = |physical_device: &back::PhysicalDevice, device: &back::Device, surface: &mut <back::Backend as hal::Backend>::Surface| {
        let (capabilities, formats, _present_modes) = surface.compatibility(physical_device);
        let new_format = formats.map_or(hal::format::Format::Rgba8Srgb, |formats| {
            formats
                .iter()
                .find(|new_format| new_format.base_format().1 == hal::format::ChannelType::Srgb)
                .map(|new_format| *new_format)
                .unwrap_or(formats[0])
        });

        let swap_config = hal::SwapchainConfig::from_caps(&capabilities, new_format);
        let new_extent = swap_config.extent.to_extent();
        let (new_swapchain, new_backbuffer) = device.create_swapchain(surface, swap_config, None);

        (new_swapchain, new_backbuffer, new_format, new_extent)
    };

    //Swapchain
    let (mut swap_chain, backbuffer, mut format, mut extent) = create_swapchain(&adapter.physical_device, &mut device, &mut surface );

    //Depth Buffer Setup
    let create_depth_buffer = |device: &back::Device, extent: &hal::image::Extent| {
        let depth_format = hal::format::Format::D32Float;
        let depth_image = device.create_image(
            hal::image::Kind::D2(extent.width as _, extent.height as _, 1, 1),
            1,
            depth_format,
            hal::image::Tiling::Optimal,
            hal::image::Usage::DEPTH_STENCIL_ATTACHMENT,
            hal::image::StorageFlags::empty(),
        ).unwrap();

        let depth_mem_reqs = device.get_image_requirements(&depth_image);

        let mem_type = memory_types
            .iter()
            .enumerate()
            .position(|(id, mem_type)| {
                depth_mem_reqs.type_mask & (1 << id) != 0 &&
                mem_type.properties.contains(hal::memory::Properties::DEVICE_LOCAL)
            })
            .unwrap()
            .into();

        let depth_memory = device.allocate_memory(mem_type, depth_mem_reqs.size).unwrap();
        let depth_image = device.bind_image_memory(&depth_memory, 0, depth_image).unwrap();

        let depth_view = device.create_image_view(
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

    let (mut depth_view, mut _depth_image, mut _depth_memory, mut depth_format) = create_depth_buffer(&device, &extent);

    //Renderpass
    let create_renderpass = |device: &back::Device, format : &hal::format::Format, depth_format: &hal::format::Format| {
        let attachment = hal::pass::Attachment {
            format: Some(*format),
            samples: 1,
            ops: hal::pass::AttachmentOps::new(
                hal::pass::AttachmentLoadOp::Clear,
                hal::pass::AttachmentStoreOp::Store,
            ),
            stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
            layouts: hal::image::Layout::Undefined..hal::image::Layout::Present,
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

        device.create_render_pass(&[attachment, depth_attachment], &[subpass], &[dependency])
    };

    let mut renderpass = create_renderpass(&device, &format, &depth_format);

    let create_framebuffers = |device: &back::Device, backbuffer: hal::Backbuffer<back::Backend>, format: &hal::format::Format, extent: &hal::image::Extent, depth_view: &<back::Backend as hal::Backend>::ImageView, renderpass: &<back::Backend as hal::Backend>::RenderPass| {
        match backbuffer {
            hal::Backbuffer::Images(images) => {
                let pairs = images
                    .into_iter()
                    .map(|image| {
                        let rtv = device.create_image_view(
                            &image, 
                            hal::image::ViewKind::D2, 
                            *format, 
                            hal::format::Swizzle::NO, 
                            COLOR_RANGE.clone(),
                            )
                            .unwrap();
                            (image, rtv)
                    })
                    .collect::<Vec<_>>();
                let fbos = pairs
                    .iter()
                    .map(|&(_, ref rtv)| {
                        device
                            .create_framebuffer(&renderpass, vec![rtv, &depth_view], *extent)
                            .unwrap()
                    })
                    .collect();
                    (pairs, fbos)
            }
            hal::Backbuffer::Framebuffer(fbo) => (Vec::new(), vec![fbo]),
        }
    };

    let (mut frame_images, mut framebuffers) = create_framebuffers(&device, backbuffer, &format, &extent, &depth_view, &renderpass);

    let create_pipeline = |device: &back::Device, renderpass: &<back::Backend as hal::Backend>::RenderPass, set_layout: &<back::Backend as hal::Backend>::DescriptorSetLayout| {
        let new_pipeline_layout = device.create_pipeline_layout(Some(set_layout), &[(hal::pso::ShaderStageFlags::VERTEX, 0..8)]);

        let new_pipeline = {
            let vs_module = {
                let glsl = fs::read_to_string("data/quad.vert").unwrap();
                let spirv: Vec<u8> = glsl_to_spirv::compile(&glsl, glsl_to_spirv::ShaderType::Vertex)
                    .unwrap()
                    .bytes()
                    .map(|b| b.unwrap())
                    .collect();
                device.create_shader_module(&spirv).unwrap()
            };
            let fs_module = {
                let glsl = fs::read_to_string("data/quad.frag").unwrap();
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
                        specialization: &[hal::pso::Specialization {
                            id: 0,
                            value: hal::pso::Constant::F32(1.0),
                        }],
                    },
                    hal::pso::EntryPoint::<back::Backend> {
                        entry: "main",
                        module: &fs_module,
                        specialization: &[],
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

                device.create_graphics_pipeline(&pipeline_desc, None)
            };

            device.destroy_shader_module(vs_module);
            device.destroy_shader_module(fs_module);

            pipeline.unwrap()
        };

        (new_pipeline, new_pipeline_layout)
    };

    let (mut pipeline, mut pipeline_layout) = create_pipeline(&device, &renderpass, &set_layout);

    let mut frame_semaphore = device.create_semaphore();
    
    let mut frame_fence = device.create_fence(false);
    let mut running = true;
    let mut needs_resize = true;
    let (mut window_width, mut window_height) = (0u32, 0u32);

    let first_timestamp = timestamp();
    let mut last_time = 0.0f64;
    let mut current_anim_time = 0.0f64;

    //TODO: Key Hashmap
    let mut w_state = false;
    let mut s_state = false;
    let mut a_state = false;
    let mut d_state = false;
    let mut e_state = false;
    let mut q_state = false;

    let mut left_mouse_down = false;

    let mut num_frames = 0;

    let mut is_fullscreen = false;

    while running {
        num_frames += 1;
        
        let mut mouse_delta = (0.0, 0.0);

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
                    winit::WindowEvent::KeyboardInput{ device_id, input } => {
                        match input.virtual_keycode {
                            Some(winit::VirtualKeyCode::W) => w_state = input.state == winit::ElementState::Pressed,
                            Some(winit::VirtualKeyCode::S) => s_state = input.state == winit::ElementState::Pressed,
                            Some(winit::VirtualKeyCode::A) => a_state = input.state == winit::ElementState::Pressed,
                            Some(winit::VirtualKeyCode::D) => d_state = input.state == winit::ElementState::Pressed,
                            Some(winit::VirtualKeyCode::E) => e_state = input.state == winit::ElementState::Pressed,
                            Some(winit::VirtualKeyCode::Q) => q_state = input.state == winit::ElementState::Pressed,
                            Some(winit::VirtualKeyCode::F) => {
                                if input.state == winit::ElementState::Pressed {
                                    is_fullscreen = !is_fullscreen;
                                    if is_fullscreen {
                                         _window.set_fullscreen(Some(_window.get_primary_monitor()));
                                    }
                                    else {
                                        _window.set_fullscreen(None);
                                    }
                                    needs_resize = true;
                                }
                            },
                            Some(winit::VirtualKeyCode::Escape) => running = false,
                            _ => {},
                        }
                    },
                    winit::WindowEvent::MouseInput { state, button, ..} => {
                        if button == winit::MouseButton::Left {
                            left_mouse_down = state == winit::ElementState::Pressed;
                        }
                    },
                    _ => (),
                }
            }
            if let winit::Event::DeviceEvent { event, .. }= event.clone() {
                match event {
                    winit::DeviceEvent::MouseMotion { delta } => {
                            mouse_delta = delta;
                    },
                    _ => {},
                }
            }
        });

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
        
        //TODO: Skip rendering when actively dragging to resizing
        if needs_resize {

            device.wait_idle().unwrap();

            //Destroy old resources

            device.destroy_graphics_pipeline(pipeline);
            device.destroy_pipeline_layout(pipeline_layout);

            for framebuffer in framebuffers {
                device.destroy_framebuffer(framebuffer);
            }

            for (_, rtv) in frame_images {
                device.destroy_image_view(rtv);
            }
            device.destroy_render_pass(renderpass);
            device.destroy_swapchain(swap_chain);

            //Build new resources         
            let (new_swapchain, new_backbuffer, new_format, new_extent) = create_swapchain(&adapter.physical_device, &mut device, &mut surface );

            swap_chain = new_swapchain;
            format = new_format;
            extent = new_extent;

            let (new_depth_view, new_depth_image, new_depth_memory, new_depth_format) = create_depth_buffer(&device, &extent);
            depth_view = new_depth_view;
            _depth_image = new_depth_image;
            _depth_memory = new_depth_memory;
            depth_format = new_depth_format;

            let new_renderpass = create_renderpass(&device, &format, &depth_format);
            renderpass = new_renderpass;

            let (new_frame_images, new_framebuffers) = create_framebuffers(&device, new_backbuffer, &format, &extent, &depth_view, &renderpass);
            frame_images = new_frame_images;
            framebuffers = new_framebuffers;

            let (new_pipeline, new_pipeline_layout) = create_pipeline(&device, &renderpass, &set_layout);
            pipeline = new_pipeline;
            pipeline_layout = new_pipeline_layout;

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

        //Rotate cam_forward & cam_up when left mouse pressed using mouse delta
        if left_mouse_down {
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

        cam_pos += move_vec;

        camera_uniform_struct.view_matrix = glm::look_at_rh(
            &cam_pos,
            &(cam_pos + cam_forward),
            &cam_up
        ).into();

        camera_uniform_struct.model_matrix = (glm::translation(&glm::vec3(0.,0., -1.))
                                              * glm::rotation(std::f32::consts::PI / 4.0, &glm::vec3(0.,1.,0.))
                                              * glm::rotation(-std::f32::consts::PI / 2.0, &glm::vec3(1.,0.,0.))).into();

        camera_uniform_struct.time = time as f32;

        {
            let mut uniform_writer = device.acquire_mapping_writer::<CameraUniform>(&uniform_buffer_memory, 0..uniform_buffer_req.size).unwrap();
            uniform_writer[0] = camera_uniform_struct; 
            device.release_mapping_writer(uniform_writer);
        }
        current_anim_time += delta_time * 1.75;

        if current_anim_time > anim_duration as f64 {
            current_anim_time = 0.0;
        }

        //Animate Bones
        for (node_index, channel) in &mut anim_channels {

            //Get Current Left & Right Keyframes
            let mut left_key_index = channel.current_left_keyframe;
            let mut right_key_index = left_key_index + 1;

            //Get those keyframe times
            let mut left_key_time = channel.keyframes.get_time(left_key_index);
            let mut right_key_time = channel.keyframes.get_time(right_key_index);

            //If anim time isn't within keyframe times, we need to increment
            while current_anim_time as f32 >= right_key_time || (current_anim_time as f32) < left_key_time {
                left_key_index = (left_key_index + 1) % channel.keyframes.len();
                right_key_index = (right_key_index + 1) % channel.keyframes.len();
                
                left_key_time = channel.keyframes.get_time(left_key_index);
                right_key_time = channel.keyframes.get_time(right_key_index);
            }

            channel.current_left_keyframe = left_key_index;

            //Lerp Value of x from a to b = (x - a) / (b - a)
            let mut lerp_value = (current_anim_time as f32 - left_key_time) / (right_key_time - left_key_time );

            if lerp_value < 0.0 { lerp_value = 0.0; }

            match &mut channel.keyframes {
                ChannelType::TranslationChannel(translations) => {
                    let left_value : glm::Vec3 = translations[left_key_index].1.into();
                    let right_value : glm::Vec3 = translations[right_key_index].1.into();
                    nodes[*node_index].translation = vec3_lerp(&left_value, &right_value, lerp_value).into();
                },
                ChannelType::RotationChannel(rotations) => {
                    let left_value = glm::Quat::from_vector(rotations[left_key_index].1.into());
                    let right_value = glm::Quat::from_vector(rotations[right_key_index].1.into());
                    nodes[*node_index].rotation = glm::quat_slerp(&left_value, &right_value, lerp_value).as_vector().clone().into();
                 },
                ChannelType::ScaleChannel(scales) => {
                    let left_value : glm::Vec3 = scales[left_key_index].1.into();
                    let right_value : glm::Vec3 = scales[right_key_index].1.into();
                    nodes[*node_index].scale = vec3_lerp(&left_value, &right_value, lerp_value).into();
                },
            }
        }

        //Now compute each matrix and upload to GPU
        for (bone_index, mut bone) in skeleton.bones.iter_mut().enumerate() {
            if let Some(node_index) = skeleton.gpu_index_to_node_index.get(&bone_index) {
                bone.joint_matrix = (skeleton.inverse_root_transform * compute_global_transform(*node_index, &nodes) * skeleton.inverse_bind_matrices[bone_index]).into();
            }
        }

        {
            let mut uniform_writer = device.acquire_mapping_writer::<GpuBone>(&skeleton_uniform_memory, 0..skeleton_uniform_req.size).unwrap();
            uniform_writer[0..skeleton.bones.len()].copy_from_slice(&skeleton.bones);
            device.release_mapping_writer(uniform_writer);
        }
        
        device.reset_fence(&frame_fence);
        command_pool.reset();

        let frame: hal::SwapImageIndex = {
            match swap_chain.acquire_image(hal::FrameSync::Semaphore(&mut frame_semaphore)) {
                Ok(i) => i,
                Err(_) => {
                    needs_resize = true;
                    println!("FAILED TO ACQUIRE IMAGE");
                    continue;
                }
            }
        };

        let submit = {
            let mut cmd_buffer = command_pool.acquire_command_buffer(false);

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
            cmd_buffer.bind_vertex_buffers(0, Some((&vertex_buffer, 0)));
            cmd_buffer.bind_index_buffer(hal::buffer::IndexBufferView {
                buffer: &index_buffer,
                offset: 0,
                index_type: hal::IndexType::U32,
            });
            cmd_buffer.bind_graphics_descriptor_sets(&pipeline_layout, 0, Some(&desc_set), &[]); //TODO

            {
                let mut encoder = cmd_buffer.begin_render_pass_inline(
                    &renderpass,
                    &framebuffers[frame as usize],
                    viewport.rect,
                    &[
                        hal::command::ClearValue::Color(hal::command::ClearColor::Float([0.8, 0.8, 0.8, 1.0,])),
                        hal::command::ClearValue::DepthStencil(hal::command::ClearDepthStencil(1.0, 0))
                    ],
                );

                encoder.draw_indexed(0..indices_vec.len() as u32, 0, 0..100);
            }

            cmd_buffer.finish()
        };

        let submission = hal::queue::Submission::new()
            .wait_on(&[(&frame_semaphore, hal::pso::PipelineStage::BOTTOM_OF_PIPE)])
            .submit(Some(submit));
        queue_group.queues[0].submit(submission, Some(&mut frame_fence));

        // TODO: replace with semaphore
        device.wait_for_fence(&frame_fence, !0);

        // present frame
        if let Err(_) = swap_chain.present(&mut queue_group.queues[0], frame, &[]) {
            needs_resize = true;
        }

        last_time = time;
    }

    let total_time = timestamp() - first_timestamp;
    println!("Avg Frame Time: {}", total_time / num_frames as f64);
}

const INITIAL_WIDTH : f64 = 1280.0;
const INITIAL_HEIGHT : f64 = 720.0;

const COLOR_RANGE: hal::image::SubresourceRange = hal::image::SubresourceRange {
    aspects: hal::format::Aspects::COLOR,
    levels: 0..1,
    layers: 0..1,
};

#[derive(Debug, Clone, Copy)]
struct Vertex {
    a_pos: [f32; 3],
    a_col: [f32; 4],
    a_uv:  [f32; 2],
    a_norm: [f32; 3],
    a_joint_indices: [f32; 4],
    a_joint_weights: [f32; 4],
}

#[derive(Debug, Clone)]
struct GpuSkeleton {
    //Flat Array of Bone Matrices (what we update and send to GPU)
    bones: Vec<GpuBone>,
    //Maps above indices to GLTF node indices (separate so that the above Vec can be copied directly to the GPU)
    gpu_index_to_node_index: HashMap<usize, usize>,
    inverse_bind_matrices: Vec<glm::Mat4>,
    inverse_root_transform: glm::Mat4,
}

impl GpuSkeleton {
    fn new() -> GpuSkeleton {
        GpuSkeleton {
            bones: Vec::new(),
            gpu_index_to_node_index: HashMap::new(),
            inverse_bind_matrices: Vec::new(),
            inverse_root_transform: glm::Mat4::identity(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct GpuBone {
    joint_matrix: [[f32;4];4],
}

#[derive(Debug, Clone)]
struct Node {
    parent: Option<usize>, //Parent Index
    children: Vec<usize>, //Children Indices
    translation: [f32; 3],
    rotation: [f32; 4],
    scale: [f32; 3],
}

impl Node {
    fn get_transform(&self) -> gltf::scene::Transform {
        gltf::scene::Transform::Decomposed {
            translation: self.translation,
            rotation: self.rotation,
            scale: self.scale,
        }
    }
}

struct AnimChannel {
    keyframes: ChannelType,
    current_left_keyframe : usize,
}

enum ChannelType {
    TranslationChannel(Vec<(f32, [f32;3])>),
    RotationChannel(Vec<(f32, [f32;4])>),
    ScaleChannel(Vec<(f32, [f32;3])>),
}

impl ChannelType {
    //Returns time value for a given index
    fn get_time(&self, index: usize) -> f32 {
        match self {
            ChannelType::TranslationChannel(t) => t[index].0,
            ChannelType::RotationChannel(r) => r[index].0,
            ChannelType::ScaleChannel(s) => s[index].0,
        }
    }

    fn len(&self) -> usize {
        match self {
            ChannelType::TranslationChannel(t) => t.len(),
            ChannelType::RotationChannel(r) => r.len(),
            ChannelType::ScaleChannel(s) => s.len(),
        }
    }
}

fn timestamp() -> f64 {
    let timespec = time::get_time();
    timespec.sec as f64 + (timespec.nsec as f64 / 1000.0 / 1000.0 / 1000.0)
}

//Helper Function to map Children to their parent nodes
fn get_node_parents(nodes : &mut gltf::iter::Nodes) -> std::collections::HashMap<usize,Option<usize>> {
    
    let mut node_parents = HashMap::new();

    fn traverse_node(node: &gltf::Node, parent: Option<usize>, node_parents : &mut std::collections::HashMap<usize,Option<usize>> ) {
        node_parents.insert(node.index(), parent);

        for child in node.children() {
            traverse_node(&child, Some(node.index()), node_parents);
        }
    }

    for node in nodes {
        traverse_node(&node, None, &mut node_parents);
    }

    node_parents
}

//Computes global transform of node at index
fn compute_global_transform(index: usize, nodes: &Vec<Node>) -> glm::Mat4 {
    
    //Build up matrix stack from node to its root
    let mut matrix_stack : Vec<glm::Mat4> = Vec::new();

    let mut current_index = index;

    let transform = nodes[current_index].get_transform();
    
    matrix_stack.push(transform.matrix().into());

    while let Some(parent_index) = nodes[current_index].parent {

        let parent_transform = nodes[parent_index].get_transform();

        matrix_stack.insert(0, parent_transform.matrix().into());

        current_index = parent_index;
    }

    let mut result = glm::Mat4::identity();

    for matrix in matrix_stack {
        result = result * matrix;
    }

    result
}

fn degrees_to_radians<T>( deg: T) -> T 
where T: num::Float {
    deg * num::cast(0.0174533).unwrap()
}

fn vec3_lerp(start: &glm::Vec3, end: &glm::Vec3, percent: f32) -> glm::Vec3 {
    start + percent * (end - start)
}