use ::hal;
use ::B;

extern crate nalgebra_glm as glm;
use std::collections::HashMap;

extern crate gltf;

use gpu_buffer::{GpuBuffer};
use ::gfx_helpers;

pub struct GltfModel {
	pub meshes 	   : Vec<Mesh>,
	pub skeletons  : Vec<Skeleton>, //TODO: allow multiple, associate with individual meshes (if not skinned, will have no skeletons)
	pub animations : Vec<Animation>,
    pub current_anim_time : f64, //TODO: animations need to live outside of skeletons (gltf animations have no concept of what skeleton they drive)
	pub nodes      : Vec<Node>,
}

//TODO: Bind correct uniform buffer for a given animated mesh
impl GltfModel {
	pub fn new( file_path : &str, device_state : &gfx_helpers::DeviceState, transfer_queue_group : &mut hal::QueueGroup<B, hal::General>) -> GltfModel {

		//Load GLTF Model
		let (gltf_model, buffers, _) = gltf::import(file_path).unwrap();

		let mut animations = Vec::new();

		for anim in gltf_model.animations() {

			let mut anim_channels = Vec::new();
			let mut anim_start_time = std::f32::MAX;
			let mut anim_end_time = 0.0;

			//Store the animation
			for channel in anim.channels() {
				//Channel Reader
				let channel_reader = channel.reader(|buffer| Some(&buffers[buffer.index()]));

				let node_index = channel.target().node().index();

				let times = channel_reader.read_inputs().unwrap();

				//TODO: clean up unwrap
				let channel_start_time = times.clone().next().unwrap();
				if channel_start_time < anim_start_time {
					anim_start_time = channel_start_time;
				}

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

				//Get Anim End Time
				for time_val in channel_reader.read_inputs().unwrap() {
				if time_val > anim_end_time { anim_end_time = time_val; }
				}
			}

			animations.push ( Animation {
				channels: anim_channels,
				start_time : anim_start_time,
				end_time: anim_end_time,
			});
		}

		//Store all nodes (their index == index in vec, parent index, children indices, and transform)
		let mut nodes = Vec::new();

		//Store Nodes
		for node in gltf_model.nodes() {

			let children_indices = node.children().map(|child| child.index()).collect::<Vec<usize>>();

			let (translation, rotation, scale) = node.transform().decomposed();

			let mut parent = None;

			//If we encounter ourselves (node) when searching children, we've found our parent
			for potential_parent in gltf_model.nodes() {
				if potential_parent.children().find( |child| child.index() == node.index()).is_some() {
					parent = Some(potential_parent.index());
				}
			}

			nodes.push(
				Node {
					parent: parent,
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

		let mut meshes = Vec::new();
        let mut skeletons = Vec::new();
        let mut skeleton_index = None;

		for node in gltf_model.nodes() {

			let has_mesh = node.mesh().is_some();
			let is_skinned = node.skin().is_some();

            //skinning: build up skeleton
			if has_mesh && is_skinned {

				match node.skin() {
					Some(skin) => {
                        let mut bones = Vec::new();
                        let mut inverse_bind_matrices = Vec::new();
                        let mut gpu_index_to_node_index = HashMap::new();

						let reader = skin.reader(|buffer| Some(&buffers[buffer.index()]));
						//If "None", then each joint's inv_bind_matrix is assumed to be 4x4 Identity matrix
						let mut gltf_inverse_bind_matrices = reader.read_inverse_bind_matrices();

						let inverse_root_transform : glm::Mat4 = glm::inverse(&node.transform().matrix().into());
						
						//Joints are nodes
						for joint in skin.joints() {

							let inverse_bind_matrix: glm::Mat4 = {

								let mut out_matrix : glm::Mat4 = glm::Mat4::identity();

								match &mut gltf_inverse_bind_matrices {
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
							let joint_transform = compute_global_transform(joint.index(), &nodes);

							let joint_matrix = inverse_root_transform * joint_transform * inverse_bind_matrix;

							bones.push(GpuBone {
								joint_matrix: joint_matrix.into(),
							});

							inverse_bind_matrices.push(inverse_bind_matrix);

							//map index
							gpu_index_to_node_index.insert(bones.len() - 1, joint.index());
						}

                        skeletons.push(Skeleton {
                            gpu_buffer : GpuBuffer::new(&bones, hal::buffer::Usage::UNIFORM, hal::memory::Properties::CPU_VISIBLE, device_state, transfer_queue_group),
                            bones : bones,
                            gpu_index_to_node_index : gpu_index_to_node_index,
                            inverse_bind_matrices : inverse_bind_matrices,
                            inverse_root_transform : inverse_root_transform,
                        });
                        skeleton_index = Some(skeletons.len() - 1);
					},
					None => {},
				}
			}

			match node.mesh() {
				Some(gltf_mesh) => {

					let mut vertices_vec = Vec::new();
					let mut indices_vec = None;

					for primitive in gltf_mesh.primitives() {

						let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
						let pos_iter = reader.read_positions().unwrap(); 
						//TODO: Better error handling if no positions (return Err("Mesh requires positions"))

						//Normals
						let mut norm_iter = reader.read_normals();

						//Optional Colors
						let mut col_iter = match reader.read_colors(0) {
							Some(col_iter) => Some(col_iter.into_rgba_f32()),
							None => None,
						};

						//Optional UVs
						let mut uv_iter = match reader.read_tex_coords(0) {
							Some(uv_iter) => Some(uv_iter.into_f32()),
							None => {
								println!("Warning: Mesh is missing UVs"); 
								None
							},
						};

						//if skinned, we need to get the JOINTS_0 and WEIGHTS_0 attributes
						let mut joints_iter = match reader.read_joints(0) {
							Some(joints_iter) => Some(joints_iter.into_u16()),
							None => {println!("NO JOINTS"); None},
						};

						let mut weights_iter = match reader.read_weights(0) {
							Some(weights_iter) => Some(weights_iter.into_f32()),
							None => {println!("NO WEIGHTS"); None},
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
						});
					} 

					meshes.push(Mesh::new(vertices_vec, indices_vec, skeleton_index, device_state, transfer_queue_group));
				},
				None => {},
			}
		}
		
		GltfModel {
			meshes 	   : meshes,
			skeletons  : skeletons,
            animations : animations,
			current_anim_time : 0.0,
			nodes 	   : nodes,
		}
	}

	pub fn animate(&mut self, skeleton_index: usize, animation_index : usize, delta_time : f64) {

		self.current_anim_time += delta_time;

        if let Some(skeleton) = &mut self.skeletons.get_mut(skeleton_index) {

            if self.current_anim_time > self.animations[animation_index].end_time as f64 {
                self.current_anim_time = self.animations[animation_index].start_time as f64;
            }

            for (node_index, channel) in &mut self.animations[animation_index].channels {

                //Get Current Left & Right Keyframes
                let mut left_key_index = channel.current_left_keyframe;
                let mut right_key_index = left_key_index + 1;

                //Get those keyframe times
                let mut left_key_time = channel.keyframes.get_time(left_key_index);
                let mut right_key_time = channel.keyframes.get_time(right_key_index);

                //FIXME: can get stuck for certain models/animations

                while self.current_anim_time as f32 >= right_key_time || (self.current_anim_time as f32) < left_key_time {
                    left_key_index = (left_key_index + 1) % channel.keyframes.len();
                    right_key_index = (right_key_index + 1) % channel.keyframes.len();
                    
                    left_key_time = channel.keyframes.get_time(left_key_index);
                    right_key_time = channel.keyframes.get_time(right_key_index);
                }

                channel.current_left_keyframe = left_key_index;

                //Lerp Value of x from a to b = (x - a) / (b - a)
                let mut lerp_value = (self.current_anim_time as f32 - left_key_time) / (right_key_time - left_key_time );

                if lerp_value < 0.0 { lerp_value = 0.0; }

                match &mut channel.keyframes {
                    ChannelType::TranslationChannel(translations) => {
                        let left_value : glm::Vec3 = translations[left_key_index].1.into();
                        let right_value : glm::Vec3 = translations[right_key_index].1.into();
                        self.nodes[*node_index].translation = glm::lerp(&left_value, &right_value, lerp_value).into();
                    },
                    ChannelType::RotationChannel(rotations) => {
                        let left_value  = glm::Quat{ coords: rotations[left_key_index].1.into() };
                        let right_value = glm::Quat{ coords: rotations[right_key_index].1.into() };
                        self.nodes[*node_index].rotation = glm::quat_slerp(&left_value, &right_value, lerp_value).as_vector().clone().into();
                    },
                    ChannelType::ScaleChannel(scales) => {
                        let left_value : glm::Vec3 = scales[left_key_index].1.into();
                        let right_value : glm::Vec3 = scales[right_key_index].1.into();
                        self.nodes[*node_index].scale = glm::lerp(&left_value, &right_value, lerp_value).into();
                    },
                }
            }

            //Now compute each matrix and upload to GPU
            for (bone_index, mut bone) in skeleton.bones.iter_mut().enumerate() {
                if let Some(node_index) = skeleton.gpu_index_to_node_index.get(&bone_index) {
                    bone.joint_matrix = (skeleton.inverse_root_transform * compute_global_transform(*node_index, &self.nodes) * skeleton.inverse_bind_matrices[bone_index]).into();
                }
            }
        }
	}

	pub fn upload_bones(&mut self, device_state : &gfx_helpers::DeviceState, transfer_queue_group : &mut hal::QueueGroup<B, hal::General>) {
        for mut skeleton in &mut self.skeletons {
		    skeleton.gpu_buffer.reupload(&skeleton.bones, device_state, transfer_queue_group);
        }
	}

	pub fn record_draw_commands( &self, encoder : &mut hal::command::RenderPassInlineEncoder<B>, instance_count : u32) {
		for mesh in &self.meshes {

            //TODO: bind correct skeleton when rendering mesh
            if let Some(skeleton_index) = &mesh.skeleton_index {
                if let Some(skeleton) = &mut self.skeletons.get(*skeleton_index) {
                    
                }
            }

			mesh.record_draw_commands(encoder, instance_count);
		}
	}

	pub fn destroy( self, device_state : &gfx_helpers::DeviceState ) {
		for mesh in self.meshes {
			mesh.destroy(device_state);
		}
	}
}

pub struct Skeleton {
    //Flat Array of Bone Matrices (what we update and send to GPU)
    pub bones: Vec<GpuBone>,
    //Maps above indices to GLTF node indices (separate so that the above Vec can be copied directly to the GPU)
    pub gpu_index_to_node_index: HashMap<usize, usize>,
    pub inverse_bind_matrices: Vec<glm::Mat4>,
    pub inverse_root_transform: glm::Mat4,
	pub gpu_buffer: GpuBuffer,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GpuBone {
    pub joint_matrix: [[f32;4];4],
}

#[derive(Debug, Clone)]
pub struct Node {
    pub parent: Option<usize>, //Parent Index
    pub children: Vec<usize>, //Children Indices
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
}

impl Node {
    pub fn get_transform(&self) -> gltf::scene::Transform {
        gltf::scene::Transform::Decomposed {
            translation: self.translation,
            rotation: self.rotation,
            scale: self.scale,
        }
    }
}

pub struct AnimChannel {
    pub keyframes: ChannelType,
    pub current_left_keyframe : usize,
}

pub enum ChannelType {
    TranslationChannel(Vec<(f32, [f32;3])>),
    RotationChannel(Vec<(f32, [f32;4])>),
    ScaleChannel(Vec<(f32, [f32;3])>),
}

impl ChannelType {
    //Returns time value for a given index
    pub fn get_time(&self, index: usize) -> f32 {
        match self {
            ChannelType::TranslationChannel(t) => t[index].0,
            ChannelType::RotationChannel(r) => r[index].0,
            ChannelType::ScaleChannel(s) => s[index].0,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            ChannelType::TranslationChannel(t) => t.len(),
            ChannelType::RotationChannel(r) => r.len(),
            ChannelType::ScaleChannel(s) => s.len(),
        }
    }
}

pub struct Animation {
    pub channels : Vec<(usize, AnimChannel)>,
	pub start_time : f32,
    pub end_time   : f32,
}

//Computes global transform of node at index
pub fn compute_global_transform(index: usize, nodes: &Vec<Node>) -> glm::Mat4 {
    
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

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Vertex {
    pub a_pos: [f32; 3],
    pub a_col: [f32; 4],
    pub a_uv:  [f32; 2],
    pub a_norm: [f32; 3],
    pub a_joint_indices: [f32; 4],
    pub a_joint_weights: [f32; 4],
}

pub struct Mesh {
    pub vertex_buffer : GpuBuffer,
    pub index_buffer : Option<GpuBuffer>,
    pub skeleton_index : Option<usize>,
}

impl Mesh {
    //TODO: replace vec arguments with slices
    pub fn new(in_vertices : Vec<Vertex>, in_indices : Option<Vec<u32>>, skeleton_index : Option<usize>, device_state : &gfx_helpers::DeviceState, transfer_queue_group : &mut hal::QueueGroup<B, hal::General> ) -> Mesh {
        Mesh {
            vertex_buffer  : GpuBuffer::new(&in_vertices, hal::buffer::Usage::VERTEX, hal::memory::Properties::DEVICE_LOCAL, device_state, transfer_queue_group),
            index_buffer   : in_indices.map(|in_indices| GpuBuffer::new(&in_indices, hal::buffer::Usage::INDEX, hal::memory::Properties::DEVICE_LOCAL, device_state, transfer_queue_group)),
            skeleton_index : skeleton_index,
        }
    }

	pub fn record_draw_commands( &self, encoder : &mut hal::command::RenderPassInlineEncoder<B>, instance_count : u32)
	{
		unsafe {
			encoder.bind_vertex_buffers(0, Some((&self.vertex_buffer.buffer, 0)));

			match &self.index_buffer {
				Some(index_buffer) => {
					encoder.bind_index_buffer(hal::buffer::IndexBufferView {
						buffer: &index_buffer.buffer,
						offset: 0,
						index_type: hal::IndexType::U32,
					});
					encoder.draw_indexed(0..index_buffer.count, 0, 0..instance_count);
				},
				None => {
					encoder.draw(0..self.vertex_buffer.count, 0..instance_count);
				}
			}
		}
	}

    pub fn destroy(self, device_state : &gfx_helpers::DeviceState) {
        self.vertex_buffer.destroy(device_state);
		match self.index_buffer {
			Some(gpu_buffer) => gpu_buffer.destroy(device_state),
			None => {},
		}
    }
}