use ::hal;
use ::B;
use ::gfx_helpers;

use ::hal::{Device, Backend};

//TODO: 2 specializations
// 1. one that uses staging buffer (requires queue group) [CPU_VISIBLE]
// 2. and one that doesn't (no queue group argument) [DEVICE LOCAL]

//TODO: Cpu Visible buffers should have a function to access data

//TODO: function that gets/sets data in buffer based on its type information (Vertex,GpuBone,etc.)

pub struct GpuBuffer {
    pub buffer : <B as Backend>::Buffer,
    pub memory : <B as Backend>::Memory,
	pub usage  : hal::buffer::Usage,
	pub memory_properties : hal::memory::Properties,
    pub buffer_reqs : hal::memory::Requirements,
    pub data_size   : u64,
    pub count       : u32,
}

impl GpuBuffer {

    //FIXME: new function per usage? 
    pub fn new_cpu_visible<T: Copy>(    data : &[T],
                            usage : hal::buffer::Usage, 
                            device_state : &gfx_helpers::DeviceState )
    -> GpuBuffer {
        let memory_properties = hal::memory::Properties::CPU_VISIBLE;

		let data_stride = std::mem::size_of::<T>() as u64;
        let data_size = data.len() as u64 * data_stride;
		
		let mut upload_buffer = unsafe { device_state.device.create_buffer(data_size, usage).unwrap() };
        let upload_buffer_req = unsafe { device_state.device.get_buffer_requirements(&upload_buffer) };

		let upload_type = gfx_helpers::get_memory_type(&device_state.physical_device, &upload_buffer_req, memory_properties);

        let upload_buffer_memory = unsafe { device_state.device.allocate_memory(upload_type, upload_buffer_req.size).unwrap() };
        unsafe { device_state.device.bind_buffer_memory(&upload_buffer_memory, 0, &mut upload_buffer).unwrap() };

        unsafe {
            let mut mapping_writer = device_state.device.acquire_mapping_writer::<T>(&upload_buffer_memory, 0..upload_buffer_req.size).unwrap();
            mapping_writer[0..data.len()].copy_from_slice(&data);
            device_state.device.release_mapping_writer(mapping_writer).unwrap();
        }

        GpuBuffer {
			buffer 			  : upload_buffer,
			memory 			  : upload_buffer_memory,
			usage  			  : usage,
			memory_properties : memory_properties,
            buffer_reqs       : upload_buffer_req,
            data_size         : data_size,
            count             : data.len() as u32,
		}
    }

	pub fn new<T: Copy>(	data : &[T],
                            usage : hal::buffer::Usage, 
                            memory_properties: hal::memory::Properties, 
                            device_state : &gfx_helpers::DeviceState,
                            transfer_queue_group : &mut hal::QueueGroup<B, hal::General> ) //TODO: make optional
	-> GpuBuffer {
        
		let use_staging_buffer = (memory_properties & hal::memory::Properties::CPU_VISIBLE) != hal::memory::Properties::CPU_VISIBLE;
		let upload_usage = if use_staging_buffer { hal::buffer::Usage::TRANSFER_SRC } else { usage };
		let upload_memory_properties = if use_staging_buffer { hal::memory::Properties::CPU_VISIBLE } else { memory_properties };

		let data_stride = std::mem::size_of::<T>() as u64;
        let data_size = data.len() as u64 * data_stride;
		
		let mut upload_buffer = unsafe { device_state.device.create_buffer(data_size, upload_usage).unwrap() };
        let upload_buffer_req = unsafe { device_state.device.get_buffer_requirements(&upload_buffer) };

		let upload_type = gfx_helpers::get_memory_type(&device_state.physical_device, &upload_buffer_req, upload_memory_properties);

        let upload_buffer_memory = unsafe { device_state.device.allocate_memory(upload_type, upload_buffer_req.size).unwrap() };
        unsafe { device_state.device.bind_buffer_memory(&upload_buffer_memory, 0, &mut upload_buffer).unwrap() };

        unsafe {
            let mut mapping_writer = device_state.device.acquire_mapping_writer::<T>(&upload_buffer_memory, 0..upload_buffer_req.size).unwrap();
            mapping_writer[0..data.len()].copy_from_slice(&data);
            device_state.device.release_mapping_writer(mapping_writer).unwrap();
        }

		if use_staging_buffer {

			let transfer_dst_usage = hal::buffer::Usage::TRANSFER_DST | usage;
			
			let mut transfer_dst_buffer = unsafe { device_state.device.create_buffer(data_size, transfer_dst_usage).unwrap() };
			let transfer_dst_buffer_req = unsafe { device_state.device.get_buffer_requirements(&transfer_dst_buffer) };

			let transfer_dst_upload_type = gfx_helpers::get_memory_type(&device_state.physical_device, &transfer_dst_buffer_req, memory_properties);

			let transfer_dst_buffer_memory = unsafe { device_state.device.allocate_memory(transfer_dst_upload_type, transfer_dst_buffer_req.size).unwrap() };
			unsafe { device_state.device.bind_buffer_memory(&transfer_dst_buffer_memory, 0, &mut transfer_dst_buffer).unwrap() };
			
			let mut command_pool = unsafe {device_state.device.create_command_pool_typed(transfer_queue_group, hal::pool::CommandPoolCreateFlags::TRANSIENT)
                            .expect("Can't create command pool") };

            let mut cmd_buffer = command_pool.acquire_command_buffer::<hal::command::OneShot>();
			unsafe {
                cmd_buffer.begin();

                cmd_buffer.copy_buffer( &upload_buffer, 
                                        &transfer_dst_buffer,
                                        &[hal::command::BufferCopy {
                                            src: 0,
                                            dst: 0,
                                            size: data_size,
                                        }]
                );

                cmd_buffer.finish();

                let mut transfer_fence = device_state.device.create_fence(false).unwrap();
                transfer_queue_group.queues[0].submit_without_semaphores(Some(&cmd_buffer), Some(&mut transfer_fence));
                device_state.device.wait_for_fence(&transfer_fence, !0).expect("Can't wait for fence");

                device_state.device.destroy_command_pool(command_pool.into_raw());

                //once transfer is done, clean up upload_buffer
                device_state.device.destroy_buffer(upload_buffer);
			    device_state.device.free_memory(upload_buffer_memory);
            }

            return GpuBuffer {
                buffer            : transfer_dst_buffer,
                memory            : transfer_dst_buffer_memory,
                usage             : transfer_dst_usage,
                memory_properties : memory_properties,
                buffer_reqs       : transfer_dst_buffer_req,
                data_size         : data_size,
                count             : data.len() as u32,
            };
		}

		GpuBuffer {
			buffer 			  : upload_buffer,
			memory 			  : upload_buffer_memory,
			usage  			  : usage,
			memory_properties : memory_properties,
            buffer_reqs       : upload_buffer_req,
            data_size         : data_size,
            count             : data.len() as u32,
		}
	}

	fn recreate<T: Copy>(&mut self, data : &[T], device_state : &gfx_helpers::DeviceState, transfer_queue_group : &mut hal::QueueGroup<B, hal::General> ) {
		*self = GpuBuffer::new(data, self.usage, self.memory_properties, device_state, transfer_queue_group);
	}

    //TODO: don't recreate buffer if there's enough space (new data is smaller than old data)
	pub fn reupload<T: Copy>(&mut self, data: &[T], device_state : &gfx_helpers::DeviceState, transfer_queue_group : &mut hal::QueueGroup<B, hal::General>) {		
        if ( data.len() * std::mem::size_of::<T>() ) as u64 != self.data_size {
			self.recreate(data, device_state, transfer_queue_group);
		} else {
			unsafe {
				let mut mapping_writer = device_state.device.acquire_mapping_writer::<T>(&self.memory, 0..self.buffer_reqs.size).unwrap();
				mapping_writer[0..data.len()].copy_from_slice(&data);
				device_state.device.release_mapping_writer(mapping_writer).unwrap();
			}
		}
	}

    //TODO: copy to vec is slow, have this function take a closure where you can temporarily access data?
    // Above would allow 1. similar to code below for CPU_VISIBLE, 2. Temporary copy to CPU_VISIBLE buffer for non-cpu-visible buffers
    pub fn get_data<T: Copy>(&self, device_state : &gfx_helpers::DeviceState) -> Vec<T> {
        unsafe {
            let mapping_reader = device_state.device.acquire_mapping_reader::<T>(&self.memory, 0..self.buffer_reqs.size)
                .expect("failed to acquire mapping reader");

            let result = mapping_reader.to_vec();

            device_state.device.release_mapping_reader(mapping_reader);

            result
		}
    }

    pub fn destroy(self, device_state : &gfx_helpers::DeviceState) {
		unsafe {
			device_state.device.destroy_buffer(self.buffer);
			device_state.device.free_memory(self.memory);
		}
    }
}