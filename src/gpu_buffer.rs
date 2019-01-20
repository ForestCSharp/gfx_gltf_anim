use ::hal;
use ::B;
use ::gfx_helpers;

use hal::{Device, Backend};

pub struct GpuBuffer {
    pub buffer : <B as Backend>::Buffer,
    pub memory : <B as Backend>::Memory,
	pub usage  : hal::buffer::Usage,
	pub memory_properties : hal::memory::Properties,
	pub count  : u32,
}

impl GpuBuffer {

	pub fn new<T : Copy>(	data : &[T], 
						 	usage : hal::buffer::Usage, 
							memory_properties: hal::memory::Properties, 
							device_state : &gfx_helpers::DeviceState,
                            transfer_queue_group : &mut hal::QueueGroup<B, hal::Transfer> ) //TODO: make optional
	-> GpuBuffer {
        
		let use_staging_buffer = (memory_properties & hal::memory::Properties::CPU_VISIBLE) != hal::memory::Properties::CPU_VISIBLE;
		let upload_usage = if use_staging_buffer { hal::buffer::Usage::TRANSFER_SRC } else { usage };
		let upload_memory_properties = if use_staging_buffer { hal::memory::Properties::CPU_VISIBLE } else { memory_properties };

		let buffer_stride = std::mem::size_of::<T>() as u64;
        let buffer_len = data.len() as u64 * buffer_stride;
		
		let mut upload_buffer = unsafe { device_state.device.create_buffer(buffer_len, upload_usage).unwrap() };
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
			
			let mut transfer_dst_buffer = unsafe { device_state.device.create_buffer(buffer_len, transfer_dst_usage).unwrap() };
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
                                            size: buffer_len,
                                        }]
                );

                cmd_buffer.finish();

                let mut transfer_fence = device_state.device.create_fence(false).unwrap();
                transfer_queue_group.queues[0].submit_nosemaphores(Some(&cmd_buffer), Some(&mut transfer_fence));
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
                count             : data.len() as u32,
            };
		}

		GpuBuffer {
			buffer 			  : upload_buffer,
			memory 			  : upload_buffer_memory,
			usage  			  : usage,
			memory_properties : memory_properties, 
			count 			  : data.len() as u32,
		}
	}

	fn recreate<T : Copy>(&mut self, data : &[T], device_state : &gfx_helpers::DeviceState, transfer_queue_group : &mut hal::QueueGroup<B, hal::Transfer> ) {
		let new_buffer = GpuBuffer::new(data, self.usage, self.memory_properties, device_state, transfer_queue_group);
		self.buffer = new_buffer.buffer;
		self.memory = new_buffer.memory;
		self.count  = data.len() as u32;
	}

	pub fn reupload<T : Copy>(&mut self, data: &[T], device_state : &gfx_helpers::DeviceState, transfer_queue_group : &mut hal::QueueGroup<B, hal::Transfer>) {
		if data.len() as u32 > self.count {
			self.recreate(data, device_state, transfer_queue_group);
		} else {
			unsafe {
				let mut mapping_writer = device_state.device.acquire_mapping_writer::<T>(&self.memory, 0..(self.count as u64 * (std::mem::size_of::<T>() as u64))).unwrap();
				mapping_writer[0..data.len()].copy_from_slice(&data);
				device_state.device.release_mapping_writer(mapping_writer).unwrap();
			}
		}
	}

    pub fn destroy(self, device_state : &gfx_helpers::DeviceState) {
		unsafe {
			device_state.device.destroy_buffer(self.buffer);
			device_state.device.free_memory(self.memory);
		}
    }
}