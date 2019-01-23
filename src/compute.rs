// Basic Compute Context

use ::B;

use hal;
use hal::{Device, Backend, DescriptorPool, CommandPool, WorkGroupCount};
use gpu_buffer::{GpuBuffer};

use gfx_helpers::DeviceState;

use std::fs;
use std::io::{Read};

//TODO: Dual-Contouring Compute Test
//Pass 1. Find and contour vertices (vertex buffer) buffer size = x*y*z
//Pass 2. Use those vertices to build faces (index buffer)
//Pass 3. Flatten those two arrays to make it easier when rendering

//Functions for:
// 1. Setting up context and describing resources
// 2. Executing compute work
// 3. Getting value of compute work (waits until finished?)

//TODO: Alternative to fences for multiple compute-passes chained together (semaphores/barriers)

//TODO: copy buffer data into non-cpu-visible working buffer and copy-back to access result

//TODO: spirv_reflect for reflection?

//TODO: storage image support?

pub struct ComputeContext {
    pub shader_module : <B as Backend>::ShaderModule,
    pub pipeline_layout : <B as Backend>::PipelineLayout,
    pub pipeline : <B as Backend>::ComputePipeline,
    pub descriptor_pool : <B as Backend>::DescriptorPool,
    pub descriptor_set_layout : <B as Backend>::DescriptorSetLayout,
    pub descriptor_set : <B as Backend>::DescriptorSet,
    pub storage_buffers : Vec<GpuBuffer>,
    pub command_pool    : hal::CommandPool<B, hal::Compute>, //TODO: allow hal::General
    pub command_buffer  : hal::command::CommandBuffer<B, hal::Compute, hal::command::MultiShot>, //TODO: allow hal::General
    pub fence           : <B as Backend>::Fence,
}

impl ComputeContext {

    pub fn new(
        shader_path : &str,
        work_group_count : WorkGroupCount,
        storage_buffers : Vec<GpuBuffer>,
        device_state : &DeviceState,
        compute_queue_group : &mut hal::QueueGroup<B, hal::Compute>,
        ) -> ComputeContext {

        let shader_module = {
            let glsl = fs::read_to_string(shader_path).expect("failed to read compute shader code to string");
            let spirv: Vec<u8> = glsl_to_spirv::compile(&glsl, glsl_to_spirv::ShaderType::Compute)
                .expect("failed to compile compute shader glsl to spirv")
                .bytes()
                .map(|b| b.expect("failure reading byte in compute shader spirv"))
                .collect();
            unsafe {
                device_state.device.create_shader_module(&spirv).expect("failed to create compute shader module")
            }
        };

        let mut layout_bindings = Vec::new();

        //Each Storage buffer gets its own layout binding 
        for i in 0..storage_buffers.len() as u32 {
            layout_bindings.push(hal::pso::DescriptorSetLayoutBinding {
                binding            : i as u32,
                ty                 : hal::pso::DescriptorType::StorageBuffer,
                count              : 1,
                stage_flags        : hal::pso::ShaderStageFlags::COMPUTE,
                immutable_samplers : false,
            });
        }

        let descriptor_set_layout = unsafe { device_state.device.create_descriptor_set_layout(
            &layout_bindings,
            &[],
        )}.expect("failed to create compute descriptor set layout");
        
        let pipeline_layout = unsafe { 
            device_state.device.create_pipeline_layout(Some(&descriptor_set_layout), &[]) 
        }.expect("failed to create compute pipeline layout");

        let pipeline = unsafe {
            let entry_point = hal::pso::EntryPoint::<B> {
                entry  : "main",
                module : &shader_module,
                specialization : hal::pso::Specialization::default(),
            };

            device_state.device.create_compute_pipeline(
                &hal::pso::ComputePipelineDesc::new(entry_point, &pipeline_layout),
                None,
            )
        }.expect("failed to create compute pipeline");

        let mut descriptor_pool = unsafe {
            device_state.device.create_descriptor_pool(
                1,
                &[hal::pso::DescriptorRangeDesc {
                    ty    : hal::pso::DescriptorType::StorageBuffer,
                    count : storage_buffers.len(),
                }],
            )
        }.expect("failed to create compute descriptor pool");

        let descriptor_set = unsafe {
            descriptor_pool.allocate_set(&descriptor_set_layout)
        }.expect("Failed to allocate compute descriptor set");

        unsafe { 
            let mut descriptor_set_writes = Vec::new();

            for i in 0..storage_buffers.len() {
                descriptor_set_writes.push(
                    hal::pso::DescriptorSetWrite {
                        set: &descriptor_set,
                        binding : i as u32,
                        array_offset : 0,
                        descriptors: Some(hal::pso::Descriptor::Buffer(
                            &storage_buffers[i].buffer, 
                            None..None)
                        ),
                    }
                );
            }

            device_state.device.write_descriptor_sets(descriptor_set_writes);
        }

        let mut command_pool = unsafe {
            device_state.device.create_command_pool_typed::<hal::Compute>(compute_queue_group, hal::pool::CommandPoolCreateFlags::empty())
        }.expect("failed to create compute command pool");

        let mut command_buffer = command_pool.acquire_command_buffer::<hal::command::MultiShot>();

        unsafe {
            //TODO: Staging->Working Copy & Working->Staging copy (so we can read results)
            /*
                Reqs: 
                    1. copy_buffer, 
                    2. transfer to compute barrier 
                    ... (bind, dispatch) ... 
                    3. compute to transfer barrier, 
                    4. coppy buffer 
            */
            
            //TODO: allow simultaneous use?
            command_buffer.begin(false);
            command_buffer.bind_compute_pipeline(&pipeline);
            command_buffer.bind_compute_descriptor_sets(&pipeline_layout, 0, Some(&descriptor_set), &[]);
            command_buffer.dispatch(work_group_count);
            command_buffer.finish();
        }

        let mut fence = device_state.device.create_fence(false).expect("failed to create compute fence");

        ComputeContext {
            shader_module         : shader_module,
            pipeline_layout       : pipeline_layout,
            pipeline              : pipeline,
            descriptor_pool       : descriptor_pool,
            descriptor_set_layout : descriptor_set_layout,
            descriptor_set        : descriptor_set,
            storage_buffers       : storage_buffers,
            command_pool          : command_pool,
            command_buffer        : command_buffer,
            fence                 : fence,
        }
    }

    pub fn dispatch(&self, compute_queue_group : &mut hal::QueueGroup<B, hal::Compute>) {

        //TODO: Don't allow dispatch if already dispatched and waiting for result

        unsafe {
            compute_queue_group.queues[0].submit_nosemaphores(Some(&self.command_buffer), Some(&self.fence));
        }
    }

    //TODO: Function that waits for fence and gets data

    pub fn destroy(self, device_state : &DeviceState) {
        unsafe {
            device_state.device.destroy_command_pool(self.command_pool.into_raw());
            device_state.device.destroy_descriptor_pool(self.descriptor_pool);
            device_state.device.destroy_descriptor_set_layout(self.descriptor_set_layout);
            device_state.device.destroy_shader_module(self.shader_module);
            device_state.device.destroy_fence(self.fence);
            device_state.device.destroy_pipeline_layout(self.pipeline_layout);
            device_state.device.destroy_compute_pipeline(self.pipeline);

            //Buffer we passed in are still considered valid
        }
    }
}