//! # GPU Management Module
//!
//! The `gpu` module is responsible for setting up and managing GPU-related resources and configurations required for rendering in a graphics application. 
//!
//! This module defines the `Gpu` struct, which encapsulates essential GPU components like the surface, device, and queue, and provides utility methods for resizing and creating GPU-dependent resources.
//!
//! ## Overview
//!
//! The module's primary contribution is the `Gpu` struct, designed to streamline GPU management. It abstracts away the complexities of initializing GPU resources, configuring the rendering surface, and handling resizing events. The utility methods offered by `Gpu` allow for a clean and efficient interaction with the GPU pipeline.
//!
//! ## Features
//!
//! - **Dynamic Surface Resizing**: Update the rendering surface dimensions and configuration dynamically.
//! - **Aspect Ratio Calculation**: Retrieve the aspect ratio of the rendering surface for content scaling.
//! - **Depth Texture Creation**: Create depth textures needed for various rendering techniques.
//! - **Asynchronous Initialization**: Enables initializing GPU resources asynchronously for better responsiveness in applications.
//!
//! ## Example Usage
//!
//! ```rust
//! use winit::window::Window;
//! use gpu::Gpu;
//!
//! async fn create_gpu_instance(window: Window, width: u32, height: u32) -> Gpu {
//!     Gpu::new_async(window, width, height).await
//! }
//! ```
//!
//! The example above demonstrates the instantiation of the `Gpu` struct for rendering to a specific window with dimensions.
//!
//! ## Module Contents
//!
//! - `Gpu`: Represents the primary struct for GPU management and operations.
//! - Utility methods for resizing, creating textures, and retrieving rendering surface properties.
//!
//! For detailed usage, see the documentation of the `Gpu` struct and its methods.

// Importing `wasm_bindgen::prelude::*` is necessary for enabling WebAssembly (WASM) compatibility.
// It provides the required traits, macros, and attributes to interact with JavaScript and the
// host environment when compiling the code for the `wasm32` target architecture.
// For example, it facilitates bindings to the browser's Web APIs and allows the communication
// between Rust and JavaScript in a WASM-based application.
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

// Importing `wgpu::InstanceDescriptor` is necessary for configuring and managing the `wgpu` instance.
// It allows you to describe the features and limits of the GPU instance, such as backends and power preferences.
// This descriptor is typically used when initializing `wgpu::Instance`, enabling customization of how the GPU interacts with the system.
use wgpu::InstanceDescriptor;

/// A struct representing the GPU-related resources and configurations required for rendering.
///
/// This struct manages the GPU surface, device, queue, surface configuration, and provides utility methods
/// for resizing, creating textures, and handling aspect ratios.
///
/// # Fields
/// - `surface` (`wgpu::Surface`): Represents the surface associated with the GPU rendering target, typically
///   tied to a window or canvas.
/// - `device` (`wgpu::Device`): The device instance that is used to create GPU-dependent resources and execute commands.
/// - `queue` (`wgpu::Queue`): The command queue, used to submit command buffers to the GPU for execution.
/// - `surface_config` (`wgpu::SurfaceConfiguration`): The configuration settings for the rendering surface,
///   such as its size, format, and other parameters.
/// - `surface_format` (`wgpu::TextureFormat`): The texture format used by the surface, obtained from the surface's capabilities.
///
/// # Methods
/// - `aspect_ratio() -> f32`: Computes the aspect ratio of the rendering surface based on the current width and height.
/// - `resize(width: u32, height: u32)`: Resizes the rendering surface to the specified dimensions and updates its configuration.
/// - `create_depth_texture(width: u32, height: u32) -> wgpu::TextureView`: Creates and returns a depth texture
///   for use in rendering, based on the specified dimensions.
/// - `new_async(window, width, height) -> Self`: Asynchronously initializes a `Gpu` instance with the specified
///   window and dimensions.
///
/// # Example
/// ```rust
/// use wgpu::Surface;
///
/// // Create a Gpu instance using a window reference and dimensions.
/// async fn create_gpu(window: winit::window::Window, width: u32, height: u32) -> Gpu {
///     Gpu::new_async(window, width, height).await
/// }
/// ```
pub struct Gpu {
    /// The surface associated with the GPU rendering target.
    ///
    /// This represents the platform-specific surface on which rendering operations
    /// are performed. Typically, it is tied to a window or a canvas, allowing the
    /// rendered output to be displayed to the user.
    ///
    /// The surface is used to configure the swap chain and ensures that rendered
    /// frames are presented on the screen.
    pub surface: wgpu::Surface<'static>,

    /// The device instance that is used to create GPU-dependent resources and execute commands.
    ///
    /// This is a handle to the graphics device provided by the system's GPU.
    /// It is responsible for resource creation (e.g., buffers, textures, and pipelines)
    /// as well as issuing command submissions to the GPU queue for execution.
    ///
    /// The `device` is obtained from a suitable adapter during initialization and
    /// exposed here to allow interaction with the GPU for rendering and computation tasks.
    /// It is essential for managing the lifetime and scope of GPU resources.
    pub device: wgpu::Device,

    /// The command queue, used to submit command buffers to the GPU for execution.
    ///
    /// This queue handles the actual execution of GPU commands by dispatching
    /// command buffers to the GPU for processing. It serves as a bridge between
    /// CPU-side operations and the GPU.
    ///
    /// It is a critical part of the rendering pipeline, enabling developers to perform
    /// draw calls, resource uploads, and synchronization tasks necessary for
    /// rendering and computations.
    ///
    /// The queue is created alongside the device and used throughout the lifecycle
    /// of rendering to ensure commands are executed efficiently.
    pub queue: wgpu::Queue,

    /// The configuration settings for the rendering surface.
    ///
    /// This field holds an instance of `wgpu::SurfaceConfiguration`, which defines
    /// how the GPU surface is configured for rendering. Configuration includes
    /// parameters such as the width, height, texture format, and other pertinent
    /// details required for proper operation of the surface.
    ///
    /// The surface configuration is essential for ensuring compatibility between
    /// the rendering surface and the device. It is used for resizing the surface
    /// and reconfiguring it when the window is resized or other configuration
    /// changes are necessary.
    ///
    /// Typical usage involves providing this configuration during surface creation
    /// or dynamically updating it during runtime to accommodate varying system
    /// conditions or user interactions.
    pub surface_config: wgpu::SurfaceConfiguration,

    /// The texture format of the rendering surface.
    ///
    /// This field specifies the texture format (`wgpu::TextureFormat`) used for the surface,
    /// which determines the color format and data layout of the rendered output.
    ///
    /// The texture format is typically selected based on the surface's capabilities
    /// and represents the format in which rendered frames are stored before being
    /// presented to the screen.
    ///
    /// It is used when configuring the surface and is essential for ensuring compatibility
    /// between the surface and the rendering pipeline.
    ///
    /// Common formats include `Bgra8Unorm` and `Rgba8Unorm`, depending on the system and
    /// rendering requirements.
    pub surface_format: wgpu::TextureFormat,
}

/// Implementation block for the `Gpu` struct, providing utility functions
/// to work with the GPU, such as querying the display's aspect ratio,
/// resizing the rendering surface, and creating GPU-dependent resources.
impl Gpu {
    /// Calculates the aspect ratio of the rendering surface.
    ///
    /// The aspect ratio is determined by dividing the width of the surface by its height.
    /// This value is useful for correctly scaling rendered content to fit the target display.
    ///
    /// # Returns
    ///
    /// A `f32` value representing the aspect ratio of the surface.
    /// If the height of the surface is zero, the function ensures no division by zero
    /// by using `1` as the minimum value for height in the calculation.
    ///
    /// # Examples
    ///
    /// Assuming the surface width is 1920 and height is 1080:
    ///
    /// ```
    /// let aspect_ratio = gpu.aspect_ratio();
    /// assert_eq!(aspect_ratio, 1920.0 / 1080.0);
    /// ```
    pub fn aspect_ratio(&self) -> f32 {
        self.surface_config.width as f32 / self.surface_config.height.max(1) as f32
    }

    /// Resizes the rendering surface to the specified dimensions.
    ///
    /// This method updates the `surface_config` of the GPU to match the new width
    /// and height values and reconfigures the rendering surface accordingly.
    ///
    /// It is typically used when the window is resized or when the rendering surface
    /// requires reconfiguration during runtime.
    ///
    /// # Parameters
    ///
    /// - `width`: The new width of the rendering surface, in pixels.
    /// - `height`: The new height of the rendering surface, in pixels.
    ///
    /// # Example
    ///
    /// Assume the rendering surface needs to be resized to 1280x720:
    ///
    /// ```
    /// gpu.resize(1280, 720);
    /// ```
    pub fn resize(&mut self, width: u32, height: u32) {
        self.surface_config.width = width;
        self.surface_config.height = height;
        self.surface.configure(&self.device, &self.surface_config);
    }

    /// Creates a depth texture for the GPU rendering pipeline.
    ///
    /// The depth texture is used for managing depth testing during rendering,
    /// ensuring that objects closer to the camera obscure ones further away.
    ///
    /// This method creates a `wgpu::Texture` configured for depth rendering,
    /// and returns its corresponding `wgpu::TextureView` for further use.
    ///
    /// # Parameters
    ///
    /// - `width`: The width of the depth texture, in pixels.
    /// - `height`: The height of the depth texture, in pixels.
    ///
    /// # Returns
    ///
    /// A `wgpu::TextureView` representing the depth texture view, which can
    /// be used as the depth attachment in a render pass.
    ///
    /// # Remarks
    ///
    /// - The depth texture uses the `Depth32Float` format, which stores a 32-bit
    ///   floating-point value for depth information.
    /// - It supports `RENDER_ATTACHMENT` and `TEXTURE_BINDING` usages, making it
    ///   suitable for rendering and sampling.
    ///
    /// # Examples
    ///
    /// ```
    /// let depth_texture = gpu.create_depth_texture(1920, 1080);
    /// ```
    pub fn create_depth_texture(&self, width: u32, height: u32) -> wgpu::TextureView {
        // The `texture` variable represents the GPU resource for the depth texture.
        //
        // It is created using the `create_texture` method, which defines the texture's
        // properties such as its dimensions, format, and usage. In this context, the
        // texture is specifically configured for depth rendering.
        //
        // # Properties
        //
        // - **Label**: The texture is labeled as "Depth Texture" for debugging purposes.
        // - **Size**: The texture has a width and height defined by the method parameters,
        //   with a depth of 1 since it is a 2D texture.
        // - **Format**: The format is set to `Depth32Float`, which provides high precision
        //   for depth calculations.
        // - **Usage**: The texture supports `RENDER_ATTACHMENT` (used in the rendering pipeline)
        //   and `TEXTURE_BINDING` (allowing it to be sampled if necessary).
        //
        // # Remarks
        //
        // The `texture` variable is central to enabling proper depth testing during rendering,
        // ensuring the visual correctness of overlapping objects in the scene.
        let texture = self.device.create_texture(
            &(wgpu::TextureDescriptor {
                label: Some("Depth Texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            }),
        );
        texture.create_view(&wgpu::TextureViewDescriptor {
            label: None,
            format: Some(wgpu::TextureFormat::Depth32Float),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            base_array_layer: 0,
            array_layer_count: None,
            mip_level_count: None,
            usage: None,
        })
    }

    /// Creates a new GPU context asynchronously.
    ///
    /// This method initializes the GPU context using the given window and dimensions.
    /// It configures the surface, requests an adapter, and sets up a compatible device and queue.
    ///
    /// This function is `async` because it relies on asynchronous GPU resource requests.
    ///
    /// # Parameters
    ///
    /// - `window`: The target window for rendering. This is used to create the GPU surface.
    /// - `width`: The initial width of the rendering surface, in pixels.
    /// - `height`: The initial height of the rendering surface, in pixels.
    ///
    /// # Returns
    ///
    /// Returns an instance of `Self` containing the configured GPU context, including
    /// the `surface`, `device`, `queue`, and `surface_config`.
    ///
    /// # Remarks
    ///
    /// - The method selects an adapter that is compatible with the provided window surface.
    /// - It ensures the surface configuration matches the surface's capabilities, selecting
    ///   a non-sRGB format for compatibility with `egui`.
    /// - Errors are logged via the `log` crate if any GPU device or adapter setup fails.
    ///
    /// # Errors
    ///
    /// Panics if it fails to create a surface, request an adapter, or request a device.
    ///
    /// # Examples
    ///
    /// ```
    /// let gpu_context = GpuContext::new_async(window, 1920, 1080).await;
    /// ```
    pub async fn new_async(
        window: impl Into<wgpu::SurfaceTarget<'static>>,
        width: u32,
        height: u32,
    ) -> Self {
        // The `instance` variable represents a handle to the WGPU instance,
        // which is the entry point for interacting with the GPU.
        //
        // # Remarks
        //
        // - The `Instance` is used to create GPU surfaces and query available adapters.
        // - It serves as the foundational object for setting up GPU-related resources.
        // - A single `Instance` can manage multiple surfaces and adapters.
        let instance = wgpu::Instance::new(&InstanceDescriptor::default());

        // The `surface` variable represents the rendering surface associated with the given window.
        //
        // # Remarks
        //
        // - The surface is created using the `Instance` and is tied to the windowing system.
        // - It serves as the GPU's target for presenting rendered frames.
        // - This surface will be configured later with a suitable format and dimensions
        //   based on the adapter’s capabilities.
        //
        // - The `Surface` is essential for rendering in windowed applications, as it allows
        //   the GPU to directly render content to the display.
        let surface = instance.create_surface(window).unwrap();

        // Represents a handle to the GPU adapter used for device creation.
        //
        // # Remarks
        //
        // - The `adapter` is selected based on the provided options, such as power preference
        //   and compatibility with the rendering surface.
        // - The `adapter` provides information about the GPU and its capabilities, such as
        //   supported features, limits, and formats.
        // - It plays a central role in determining whether the system's GPU meets the
        //   requirements for rendering and computation tasks.
        //
        // # Error Handling
        //
        // - If no suitable adapter is found that matches the provided options, the operation
        //   will fail, causing the application to panic with an appropriate error message.
        //
        // # Example Usage
        //
        // ```rust
        // let adapter = instance
        //     .request_adapter(&wgpu::RequestAdapterOptions {
        //         power_preference: wgpu::PowerPreference::HighPerformance,
        //         compatible_surface: Some(&surface),
        //         force_fallback_adapter: false,
        //     })
        //     .await
        //     .expect("Failed to request adapter!");
        // ```
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to request adapter!");

        // Represents the GPU device used for rendering and computation.
        //
        // # Remarks
        //
        // - The `device` is created from the selected GPU adapter and serves as the primary
        //   interface for interacting with the GPU.
        // - It allows the creation of various resources such as buffers, textures, and pipelines,
        //   which are essential for rendering and compute tasks.
        // - The `device` also manages the underlying GPU's state and ensures efficient resource usage.
        //
        // # Importance
        //
        // - The `device` is a central component in the WGPU pipeline, enabling the execution of
        //   GPU-based operations.
        // - With the `queue`, it forms the foundation for submitting commands to the GPU.
        //
        // # Error Handling
        //
        // - If the GPU device fails to be created due to invalid configurations or hardware issues,
        //   the operation will fail, and the application will panic with an error message.
        //
        // ---
        //
        // Represents the command queue associated with the GPU device.
        //
        // # Remarks
        //
        // - The `queue` is responsible for submitting command buffers to the GPU for execution.
        // - It provides an interface for queuing up and executing GPU commands related to rendering and computation.
        // - The `queue` works in conjunction with the `device` to facilitate the rendering pipeline.
        //
        // # Importance
        //
        // - The `queue` plays a critical role in orchestrating the GPU's workflow by scheduling and
        //   managing the execution of tasks.
        // - It allows the application to send rendering commands or any GPU workload efficiently.
        //
        // # Error Handling
        //
        // - If the `queue` fails to process commands due to invalid configurations or resource limitations,
        //   it may lead to rendering errors or interruptions in the application's graphical output.
        let (device, queue) = {
            log::info!("WGPU Adapter Features: {:#?}", adapter.features());
            adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("WGPU Device"),
                        memory_hints: wgpu::MemoryHints::default(),
                        required_features: wgpu::Features::default(),
                        #[cfg(not(target_arch = "wasm32"))]
                        required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
                        #[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
                        required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
                        #[cfg(all(target_arch = "wasm32", feature = "webgl"))]
                        required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                            .using_resolution(adapter.limits()),
                    },
                    None,
                )
                .await
                .expect("Failed to request a device!")
        };

        // Represents the capabilities of the surface as determined by the selected GPU adapter.
        //
        // # Remarks
        //
        // - The `surface_capabilities` contain information about what surface configurations are supported
        //   by the GPU and the display system.
        // - These capabilities include supported texture formats, presentation modes, and alpha compositing modes.
        //
        // # Fields
        //
        // - `formats`: A list of `wgpu::TextureFormat` specifying which texture formats are supported by the surface.
        // - `present_modes`: A list of `wgpu::PresentMode` enumerating the possible methods of presenting the surface's textures
        //   (e.g., V-Sync, Mailbox, Immediate, etc.).
        // - `alpha_modes`: A list of `wgpu::CompositeAlphaMode` that describe how the alpha channel is handled during compositing.
        //
        // # Importance
        //
        // - Knowing the surface capabilities is essential for configuring the surface correctly to match
        //   the desired performance, visual quality, and compatibility requirements.
        // - This ensures that the selected `format`, `present_mode`, and `alpha_mode` align with what the
        //   GPU and platform can support.
        //
        // # Usage Example
        //
        // ```rust
        // let surface_format = surface_capabilities
        //     .formats
        //     .iter()
        //     .copied()
        //     .find(|f| !f.is_srgb())
        //     .unwrap_or(surface_capabilities.formats[0]);
        // ```
        let surface_capabilities = surface.get_capabilities(&adapter);

        // The surface texture format selected for rendering.
        //
        // # Remarks
        //
        // - The `surface_format` specifies the format of the texture that will be used for rendering operations.
        // - It is chosen based on the capabilities of the surface, ensuring compatibility with the GPU and the rendering pipeline.
        // - A non-sRGB format is preferred by default as it aligns with the requirements of the `egui` library.
        //
        // # Purpose
        //
        // - Defines the format of the surface textures, which includes properties like color depth, color space, and compression.
        // - Ensures that the texture format matches both the application's needs and the GPU's capabilities.
        //
        // # Importance
        //
        // - The texture format impacts the quality and performance of the rendering.
        // - By selecting an appropriate format, the rendering pipeline can achieve optimal color reproduction and efficiency.
        //
        // # Error Handling
        //
        // - If no suitable format is found, it defaults to the first format in the list supported by the surface capabilities.
        //
        // # Usage Example
        //
        // ```rust
        // let surface_format = surface_capabilities
        //     .formats
        //     .iter()
        //     .copied()
        //     .find(|f| !f.is_srgb()) // egui prefers non-sRGB surface texture
        //     .unwrap_or(surface_capabilities.formats[0]);
        // ```
        let surface_format = surface_capabilities
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb()) // egui wants a non-srgb surface texture
            .unwrap_or(surface_capabilities.formats[0]);

        // The configuration settings for the surface.
        //
        // # Remarks
        //
        // - The `surface_config` defines how the surface should operate and interact with the
        //   GPU, including details like texture usage, format, dimensions, and presentation mode.
        // - Properly configuring the surface ensures that rendering can proceed efficiently
        //   and with the desired visual quality.
        //
        // # Fields
        //
        // - `usage`: Specifies how the texture will be used. The `RENDER_ATTACHMENT` usage ensures
        //   that it will be utilized for rendering operations.
        // - `format`: The selected texture format (`surface_format`) for the surface.
        // - `width`: Width of the surface in pixels.
        // - `height`: Height of the surface in pixels.
        // - `present_mode`: Determines how the surface's textures are presented. This impacts
        //   performance and visual behavior (e.g., vsync).
        // - `alpha_mode`: Specifies how the alpha channel is handled during compositing.
        // - `view_formats`: A list of additional view formats supported by the surface textures.
        // - `desired_maximum_frame_latency`: Controls the number of frames the GPU can process
        //   ahead, which affects latency and throughput.
        //
        // # Importance
        //
        // - Correct configuration is crucial for compatibility with the display system and
        //   ensuring smooth and artifact-free rendering.
        // - It allows the application to match the requirements of specific rendering pipelines
        //   and desired user experience (e.g., reduced latency or increased frame rates).
        //
        // # Usage Example
        //
        // ```rust
        // let surface_config = wgpu::SurfaceConfiguration {
        //     usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        //     format: surface_format,
        //     width,
        //     height,
        //     present_mode: surface_capabilities.present_modes[0],
        //     alpha_mode: surface_capabilities.alpha_modes[0],
        //     view_formats: vec![],
        //     desired_maximum_frame_latency: 2,
        // };
        // ```
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: surface_capabilities.present_modes[0],
            alpha_mode: surface_capabilities.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &surface_config);

        Self {
            surface,
            device,
            queue,
            surface_config,
            surface_format,
        }
    }
}
