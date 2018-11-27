#[cfg(feature = "dx12")]
extern crate gfx_backend_dx12 as back;
#[cfg(feature = "metal")]
extern crate gfx_backend_metal as back;
#[cfg(feature = "vulkan")]
extern crate gfx_backend_vulkan as back;

use back::Backend as B;

extern crate gfx_hal as hal;
use hal::{PhysicalDevice, Device, Backend};

extern crate nalgebra_glm as glm;
use std::collections::HashMap;

extern crate gltf;

use ::mesh;
use mesh::{Vertex,Mesh};

pub struct GltfModel {
	pub meshes 	 : Vec<Mesh>,
	pub skeleton : Skeleton, //TODO: Make optional, allow multiple, associate with individual meshes
	pub nodes    : Vec<Node>,
}

//TODO: Store Skeleton Gpu Resources
//TODO: Function to animate and upload to GPU
//TODO: Bind correct uniform buffer for a given animated mesh
impl GltfModel {
	pub fn new( file_path : &str, device : &back::Device, physical_device : &back::PhysicalDevice) -> GltfModel {

		let mut skeleton = Skeleton::new();

		//Load GLTF Model
		let (gltf_model, buffers, _) = gltf::import(file_path).unwrap();

		for anim in gltf_model.animations() {

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

			skeleton.animations.push ( Animation {
				channels: anim_channels,
				duration: anim_duration,
			});
		}

		//Map child indices to parent indices (used below when building up node Vec)
		let node_parents = get_node_parents(&mut gltf_model.nodes());

		//Store all nodes (their index == index in vec, parent index, children indices, and transform)
		let mut nodes = Vec::new();

		//Store Nodes
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

		let mut meshes = Vec::new();

		for node in gltf_model.nodes() {

			let has_mesh = node.mesh().is_some();
			let is_skinned = node.skin().is_some();

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

					meshes.push(Mesh::new(vertices_vec, indices_vec, device, physical_device));
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
							let joint_transform = compute_global_transform(joint.index(), &nodes);

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
		
		GltfModel {
			meshes 	 : meshes,
			skeleton : skeleton,
			nodes 	 : nodes,
		}
	}

	//TODO: remove dependency on primary command buffers
	pub fn record_draw_commands( &self, encoder : &mut hal::command::RenderPassInlineEncoder<B, hal::command::Primary>) {
		for mesh in &self.meshes {
			mesh.record_draw_commands(encoder);
		}
	}

	pub fn destroy( self, device : &back::Device ) {
		for mesh in self.meshes {
			mesh.destroy(device);
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
    pub animations: Vec<Animation>,
}

impl Skeleton {
    pub fn new() -> Skeleton {
        Skeleton {
            bones: Vec::new(),
            gpu_index_to_node_index: HashMap::new(),
            inverse_bind_matrices: Vec::new(),
            inverse_root_transform: glm::Mat4::identity(),
            animations: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
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
    pub duration : f32,
}

//Helper Function to map Children to their parent nodes
pub fn get_node_parents(nodes : &mut gltf::iter::Nodes) -> std::collections::HashMap<usize,Option<usize>> {
    
    let mut node_parents = HashMap::new();

    for node in nodes.clone() {

		//Default to "No Parent" or None
		node_parents.insert(node.index(), None);

		//If we encounter ourselves (node) when searching children, we've found our parent
		for potential_parent in nodes.clone() {
			if potential_parent.children().find( |child| child.index() == node.index()).is_some() {
				node_parents.insert(node.index(), Some(potential_parent.index()));
			}
		}
    }

    node_parents
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