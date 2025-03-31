//! # Vertex Module
//!
//! This module provides the `Vertex` struct and its associated methods for defining and managing
//! vertex data in a 3D graphics pipeline. Vertices are a fundamental building block for rendering
//! 3D objects, as they define the position and color of points in 3D space.
//!
//! The `Vertex` struct includes both positional data and color information, which is formatted
//! and passed to the GPU's vertex buffer. The GPU processes this data during rendering to produce
//! visuals on the screen.
//!
//! # Overview
//!
//! ## Structs
//!
//! - [`Vertex`]: Represents a single vertex in 3D space, including position and color attributes.
//!
//! ## Methods
//!
//! - [`Vertex::vertex_attributes`]: Returns the vertex attributes layout supported by the `Vertex` struct.
//! - [`Vertex::description`]: Returns the high-level memory layout for vertex data to be provided to the GPU.
//!
//! ## Usage
//!
//! This module is primarily used when creating and configuring a rendering pipeline that requires
//! vertex data. Developers can define vertex buffers based on the `Vertex` struct and use the
//! provided methods to define how the GPU should interpret this data.
//!
//! ```rust
//! use wgpu::util::DeviceExt;
//! use crate::vertex::{Vertex};
//!
//! let vertex_data = vec![
//!     Vertex {
//!         position: [0.0, 1.0, 0.0, 1.0],
//!         color: [1.0, 0.0, 0.0, 1.0],
//!     },
//!     Vertex {
//!         position: [-1.0, -1.0, 0.0, 1.0],
//!         color: [0.0, 1.0, 0.0, 1.0],
//!     },
//!     Vertex {
//!         position: [1.0, -1.0, 0.0, 1.0],
//!         color: [0.0, 0.0, 1.0, 1.0],
//!     },
//! ];
//!
//! // Create a GPU-compatible vertex buffer.
//! let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
//!     label: Some("Vertex Buffer"),
//!     contents: bytemuck::cast_slice(&vertex_data),
//!     usage: wgpu::BufferUsages::VERTEX,
//! });
//! ```
//!
//! # Features
//!
//! - Easy definition of vertex data with position and color attributes.
//! - Automatic generation of GPU-compatible buffer layouts through `vertex_attributes` and `description` methods.
//!
//! # GPU Compatibility
//!
//! The layouts and attributes defined in this module are tailored for GPUs that utilize the
//! [wgpu crate](https://crates.io/crates/wgpu) for rendering. Ensure that your shaders match the
//! memory layout defined here when working in larger rendering projects.
//!
//! # Crate Dependencies
//!
//! This module depends on the following crates:
//! - `wgpu` for the GPU attributes and layouts.
//! - `bytemuck` for safe and efficient conversion of structs for GPU usage.

/// Represents a single vertex in a 3D scene, including its position and color attributes.
///
/// This struct is used to define the data structure for vertices passed to the GPU
/// through vertex buffers. Each `Vertex` object contains a 4D position vector and a 4D
/// color vector, both stored as arrays of `f32`.
///
/// # Fields
///
/// - `position`: A `[f32; 4]` array representing the position of the vertex in 3D space.
///   The fourth component is typically used for homogenous coordinates in rendering pipelines.
/// - `color`: A `[f32; 4]` array representing the color of the vertex. The values are
///   typically normalized between 0.0 and 1.0, corresponding to RGBA components.
///
/// # Usage
///
/// This structure, when combined with the `Vertex` trait, assists in creating
/// vertex buffer layouts for passing data to a graphics pipeline.
///
/// ```rust
/// let vertex = Vertex {
///     position: [0.0, 1.0, 0.0, 1.0],
///     color: [1.0, 0.0, 0.0, 1.0],
/// };
/// ```
///
/// Additionally, the `vertex_attributes` and `description` methods can be used to
/// define the vertex buffer layout for the GPU.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    /// The position of the vertex in 3D space.
    ///
    /// This is a `[f32; 4]` array, where the first three elements represent the
    /// x, y, and z coordinates, and the fourth element is typically used for
    /// homogenous coordinates in 3D graphics pipelines. This allows for operations
    /// such as perspective division during rendering.
    ///
    /// # Example
    ///
    /// ```rust
    /// let vertex_position = [0.0, 1.0, 0.0, 1.0];
    /// ```
    position: [f32; 4],

    /// The color of the vertex.
    ///
    /// This is a `[f32; 4]` array, representing the RGBA (Red, Green, Blue, Alpha) components
    /// of the vertex's color. Each component is typically normalized between 0.0 and 1.0,
    /// where 0.0 represents no intensity and 1.0 represents full intensity.
    ///
    /// # Example
    ///
    /// ```rust
    /// let vertex_color = [1.0, 0.0, 0.0, 1.0]; // Red color with full opacity
    /// ```
    color: [f32; 4],
}

/// Implementation of methods for the `Vertex` struct which represents a 3D model vertex.
///
/// The `vertex_attributes` and `description` methods define how the vertex data
/// is laid out in memory for the GPU. This includes details on position and color
/// attributes supported by the vertex shader.
///
/// # Methods
///
/// - `vertex_attributes`: Generates a list of vertex attributes for this struct.
///     This defines which attributes the GPU expects and their data types.
///
/// - `description`: Returns the vertex buffer layout, which describes the memory
///     layout of the vertex buffer. It includes the stride, step mode, and
///     attributes.
///
/// # Usage
///
/// These methods are essential for preparing data to be sent to the graphics API.
///
/// ```rust
/// let attributes = Vertex::vertex_attributes();
/// let layout = Vertex::description(&attributes);
/// ```
///
/// The layout is passed during pipeline creation, while the attributes are
/// used to create shaders and bind proper data from the buffer.
impl Vertex {
    /// Generates the vertex attributes layout for the `Vertex` struct.
    ///
    /// This method defines how the vertex data is interpreted by the GPU, specifying
    /// the attributes for the `position` and `color` fields in the vertex shader.
    ///
    /// # Returns
    ///
    /// A `Vec<wgpu::VertexAttribute>` array consisting of two vertex attributes:
    ///
    /// - The first attribute corresponds to the `position` field and is represented
    ///   as a 4-component floating-point vector (`Float32x4`).
    /// - The second attribute corresponds to the `color` field and is represented
    ///   as a 4-component floating-point vector (`Float32x4`).
    ///
    /// These attributes are indexed starting from 0 in the vertex shader.
    ///
    /// # Example
    ///
    /// ```rust
    /// let attributes = Vertex::vertex_attributes();
    /// // attributes[0] will represent the layout for `position`
    /// // attributes[1] will represent the layout for `color`
    /// ```
    ///
    /// # GPU Compatibility
    ///
    /// This layout is essential for configuring how the GPU interprets vertex data
    /// passed to it during rendering pipeline setup.
    pub fn vertex_attributes() -> Vec<wgpu::VertexAttribute> {
        wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4].to_vec()
    }

    /// Returns the vertex buffer layout for the `Vertex` struct.
    ///
    /// This method defines the memory layout of vertex data for the GPU, including
    /// the stride (size of a single vertex in bytes), vertex step mode, and the attributes
    /// (e.g., position and color). The vertex buffer layout is used during pipeline creation
    /// to correctly interpret and bind vertex data.
    ///
    /// # Parameters
    ///
    /// - `attributes`: A reference to a slice of `wgpu::VertexAttribute`, which defines
    ///   the vertex attributes (such as position and color) and their data types.
    ///
    /// # Returns
    ///
    /// A `wgpu::VertexBufferLayout` struct containing:
    ///
    /// - `array_stride`: Size of a single vertex, calculated as the size of the `Vertex` struct.
    /// - `step_mode`: Defines how vertices are processed (e.g., per-vertex or per-instance).
    ///   This defaults to `wgpu::VertexStepMode::Vertex`, which processes one vertex at a time.
    /// - `attributes`: The provided slice of vertex attributes, describing how the vertex data
    ///   is laid out in memory.
    ///
    /// # Example
    ///
    /// ```rust
    /// let attributes = Vertex::vertex_attributes();
    /// let layout = Vertex::description(&attributes);
    /// // Use `layout` for configuring the vertex buffer in the GPU pipeline.
    /// ```
    ///
    /// # GPU Compatibility
    ///
    /// The returned layout is critical for setting up the rendering pipeline, ensuring that
    /// vertex data is interpreted correctly according to the defined attributes and their memory layout.
    pub fn description(attributes: &[wgpu::VertexAttribute]) -> wgpu::VertexBufferLayout {
        // Constructs a `wgpu::VertexBufferLayout` using the provided vertex attributes, defining the memory
        // layout for a single vertex, including its stride, step mode, and attributes. This layout is used to
        // inform the GPU how vertex data is structured and accessed during rendering.
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes,
        }
    }
}

/// An array of `Vertex` instances representing the vertices of a triangle.
///
/// Each vertex includes:
/// - `position`: The 4D coordinates of the vertex in homogeneous space, defined as `[x, y, z, w]`.
/// - `color`: The color of the vertex represented as `[r, g, b, a]` with channels specified in the range `[0.0, 1.0]`.
///
/// This triangle is defined in a right-handed coordinate system:
/// - The first vertex is located at `[1.0, -1.0, 0.0, 1.0]` with red color `[1.0, 0.0, 0.0, 1.0]`.
/// - The second vertex is at `[-1.0, -1.0, 0.0, 1.0]` with green color `[0.0, 1.0, 0.0, 1.0]`.
/// - The third vertex is at `[0.0, 1.0, 0.0, 1.0]` with blue color `[0.0, 0.0, 1.0, 1.0]`.
///
/// These vertices can be used to construct a basic triangle in a graphics pipeline, where each vertex
/// contributes to the shape and color of the rendered primitive.
pub const VERTICES: [Vertex; 3] = [
    Vertex {
        position: [1.0, -1.0, 0.0, 1.0],
        color: [1.0, 0.0, 0.0, 1.0],
    },
    Vertex {
        position: [-1.0, -1.0, 0.0, 1.0],
        color: [0.0, 1.0, 0.0, 1.0],
    },
    Vertex {
        position: [0.0, 1.0, 0.0, 1.0],
        color: [0.0, 0.0, 1.0, 1.0],
    },
];
