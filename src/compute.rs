// Basic Compute Context

use ::B;

use hal;
use hal::{Device, Backend, DescriptorPool, WorkGroupCount};
use gpu_buffer::{GpuBuffer};

use gfx_helpers::DeviceState;

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
    pub pipeline_layout : <B as Backend>::PipelineLayout,
    pub pipeline : <B as Backend>::ComputePipeline,
    pub descriptor_pool : <B as Backend>::DescriptorPool,
    pub descriptor_set_layout : <B as Backend>::DescriptorSetLayout,
    pub descriptor_set : <B as Backend>::DescriptorSet,
    pub buffers : Vec<GpuBuffer>,
    pub command_pool    : hal::CommandPool<B, hal::Compute>, //TODO: allow hal::General
    pub command_buffer  : hal::command::CommandBuffer<B, hal::Compute, hal::command::MultiShot>, //TODO: allow hal::General
    pub fence           : <B as Backend>::Fence,
}

impl ComputeContext {

    pub fn new(
        shader_module : &<B as Backend>::ShaderModule,
        work_group_count : WorkGroupCount,
        buffers : Vec<GpuBuffer>,
        device_state : &DeviceState,
        compute_queue_group : &hal::QueueGroup<B, hal::Compute>,
        ) -> ComputeContext {

        let mut layout_bindings = Vec::new();

        //Each Storage buffer gets its own layout binding
        for i in 0..buffers.len() {
            //FIXME: check that it contains correct usage, not exact match
            let buffer_type = match buffers[i].usage {
                hal::buffer::Usage::STORAGE => hal::pso::DescriptorType::StorageBuffer,
                hal::buffer::Usage::UNIFORM => hal::pso::DescriptorType::UniformBuffer,
                _ => panic!("Unsupported Buffer of usage {:?} given to compute context", buffers[i].usage), 
                //TODO: return Error instead of panic
            };

            layout_bindings.push(hal::pso::DescriptorSetLayoutBinding {
                binding            : i as u32,
                ty                 : buffer_type,
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

        //Create descriptor pool with enough space for our storage and uniform buffers
        let mut descriptor_pool = unsafe {
            device_state.device.create_descriptor_pool(
                1,
                &[hal::pso::DescriptorRangeDesc {
                    ty    : hal::pso::DescriptorType::StorageBuffer,
                    count : buffers.iter().filter(|&b| b.usage == hal::buffer::Usage::STORAGE).count(),
                },
                hal::pso::DescriptorRangeDesc {
                    ty    : hal::pso::DescriptorType::UniformBuffer,
                    count : buffers.iter().filter(|&b| b.usage == hal::buffer::Usage::UNIFORM).count(),
                }],
                hal::pso::DescriptorPoolCreateFlags::empty()
            )
        }.expect("failed to create compute descriptor pool");

        let descriptor_set = unsafe {
            descriptor_pool.allocate_set(&descriptor_set_layout)
        }.expect("Failed to allocate compute descriptor set");

        unsafe { 
            let mut descriptor_set_writes = Vec::new();

            for i in 0..buffers.len() {
                descriptor_set_writes.push(
                    hal::pso::DescriptorSetWrite {
                        set: &descriptor_set,
                        binding : i as u32,
                        array_offset : 0,
                        descriptors: Some(hal::pso::Descriptor::Buffer(
                            &buffers[i].buffer, 
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
            //TODO: allow simultaneous use?
            command_buffer.begin(false);
            command_buffer.bind_compute_pipeline(&pipeline);
            command_buffer.bind_compute_descriptor_sets(&pipeline_layout, 0, Some(&descriptor_set), &[]);
            command_buffer.dispatch(work_group_count);
            command_buffer.finish();
        }

        let fence = device_state.device.create_fence(false).expect("failed to create compute fence");

        ComputeContext {
            pipeline_layout       : pipeline_layout,
            pipeline              : pipeline,
            descriptor_pool       : descriptor_pool,
            descriptor_set_layout : descriptor_set_layout,
            descriptor_set        : descriptor_set,
            buffers               : buffers,
            command_pool          : command_pool,
            command_buffer        : command_buffer,
            fence                 : fence,
        }
    }

    pub fn dispatch(&self, queue : &mut hal::CommandQueue<B, hal::Compute>) {

        //TODO: Don't allow dispatch if already dispatched and waiting for result

        unsafe {
            queue.submit_nosemaphores(Some(&self.command_buffer), Some(&self.fence));
        }
    }

    pub fn wait_for_completion(&self, device_state : &DeviceState)
    {
        unsafe {
            //TODO: remove expect and handle error another way?
            device_state.device.wait_for_fence(&self.fence, !0).expect("Failed to wait for fence");
            device_state.device.reset_fence(&self.fence).expect("Failed to reset fence");
        }
    }

    pub fn destroy(self, device_state : &DeviceState, destroy_buffers : bool) {
        unsafe {
            device_state.device.destroy_command_pool(self.command_pool.into_raw());
            device_state.device.destroy_descriptor_pool(self.descriptor_pool);
            device_state.device.destroy_descriptor_set_layout(self.descriptor_set_layout);
            device_state.device.destroy_fence(self.fence);
            device_state.device.destroy_pipeline_layout(self.pipeline_layout);
            device_state.device.destroy_compute_pipeline(self.pipeline);

            if destroy_buffers {
                for buffer in self.buffers {
                    buffer.destroy(device_state);
                }
            }
        }
    }
}