//! # `renderer.rs` - Renderer Module
//!
//! The `renderer.rs` module contains the definition and implementation of the `Renderer` struct,
//! which provides the backbone for rendering both graphical 3D content and GUI elements in the application.
//!
//! This module integrates the `wgpu` framework for high-performance 3D rendering and the `egui_wgpu` library
//! for GUI processing. By combining both rendering techniques into a cohesive pipeline, it ensures 
//! smooth real-time rendering performance and seamless integration of interactive user interfaces with
//! application-specific 3D scenes.
//!
//! ## Key Features
//!
//! - **GPU Management**: The `Renderer` manages all resources required for GPU operations, including
//!   device initialization, command queues, surface configurations, and texture management.
//!
//! - **Depth Buffering**: High-precision depth textures (`Depth32Float`) are used to calculate and
//!   ensure proper object occlusion and realistic visual output in 3D rendering.
//!
//! - **GUI Rendering**: The `Renderer` provides an `egui_wgpu::Renderer` instance for handling rich GUI
//!   interactions and visuals, transforming user interface commands into draw calls processed by the GPU.
//!
//! - **Scene Integration**: It connects with the `Scene` struct, which encapsulates and updates 3D objects,
//!   lighting, transformations, and animations, ensuring efficient resource usage and realistic rendering.
//!
//! ## Components
//!
//! - **`Renderer` Struct**: 
//!   - Orchestrates the rendering process by maintaining all graphical resources and executing rendering
//!     commands during the application's runtime.
//!   - Fields include GPU resources (`Gpu`), depth textures, `egui` renderer, and the `Scene` object.
//!
//! - **Constants**: 
//!   - `DEPTH_FORMAT`: Defines the texture format for the depth buffer, ensuring proper depth testing for 3D scenes.
//!
//! - **Methods**: 
//!   - `new`: Initializes the renderer and allocates necessary GPU resources.
//!   - `resize`: Rescales the rendering resources when the window or surface size changes.
//!   - `render_frame`: Coordinates 3D and GUI rendering into a single, composite frame for display.
//!
//! ## Usage
//!
//! The `Renderer` is the central rendering component of the application. It is designed to be invoked
//! in the application's main rendering loop. The flow typically involves:
//! 1. Updating the `Scene` to reflect state changes, such as user input, animations, or physics.
//! 2. Processing GUI commands issued by `egui` and rendering graphical elements as a texture.
//! 3. Rendering the 3D scene and combining it with GUI textures for the final display output.
//!
//! ## Example
//!
//! ```rust
//! async fn run_app() {
//!     // Initialize the window and rendering surface
//!     let (window, width, height) = initialize_window();
//!
//!     // Create the Renderer
//!     let mut renderer = Renderer::new(&window, width, height).await.unwrap();
//!
//!     // Main render loop
//!     loop {
//!         // Handle window events and resize if necessary
//!         window.handle_events(|event| {
//!             if is_resize_event(&event) {
//!                 renderer.resize(new_width, new_height);
//!             }
//!         });
//!
//!         // Update and render the frame
//!         renderer.render_frame();
//!     }
//! }
//! ```

// Imports the `Gpu` struct from the `gpu` module, which provides GPU-related resources 
// and utilities necessary for managing the rendering process in the `Renderer`.
use crate::gpu::Gpu;

// Brings the `Scene` struct into scope, which represents a 3D scene containing
// the model, buffers, and rendering pipeline configuration. It is used within
// the `Renderer` to manage the 3D content and transformations during rendering.
use crate::scene::Scene;

/// The `Renderer` struct is responsible for rendering the application's graphical content,
/// including the 3D scene and GUI, using the `wgpu` and `egui_wgpu` frameworks.
///
/// It provides functionality to manage GPU resources, render pipelines, and textures
/// for both graphical and GUI elements. The `Renderer` struct integrates the GUI managed by
/// `egui` with the 3D rendering provided by the application-specific scene.
///
/// # Fields
///
/// - `gpu`: A wrapper around WGPU-related resources, responsible for managing the `wgpu` device,
///   queue, and surface configuration.
/// - `depth_texture_view`: A depth texture view created for rendering 3D content.
///   Uses `Depth32Float` format for depth calculations.
/// - `egui_renderer`: A renderer instance for rendering GUI elements created with `egui`.
/// - `scene`: The application's 3D scene, handling objects, transformations, and updates.
///
/// # Methods
///
/// - `new`: An asynchronous function that initializes the `Renderer` with the necessary GPU
///   resources, depth texture, GUI renderer, and scene data.
/// - `resize`: Resizes the renderer's resources when the window or surface size changes.
/// - `render_frame`: Renders a frame by updating the scene, processing GUI commands,
///   and combining all drawing operations into a final image for display.
///
/// # Usage
///
/// The `Renderer` is typically used within an application loop to handle real-time rendering
/// of both the GUI and the 3D graphics. It integrates `wgpu`'s rendering pipeline with `egui`'s
/// GUI commands, ensuring seamless interaction between the two.
pub struct Renderer {
    /// A wrapper around WGPU-related resources, responsible for managing
    /// the GPU's device, queue, and surface configuration.
    ///
    /// The `Gpu` struct handles the initialization and lifecycle of WGPU
    /// components such as the rendering device, command queue, and the surface
    /// which is used to present rendered frames. It also provides helper functions
    /// for creating textures and managing GPU-specific resources.
    gpu: Gpu,

    /// A depth texture view created for rendering 3D content.
    ///
    /// This texture view is used during the rendering process to store depth
    /// information for 3D scenes. It enables proper rendering of objects
    /// based on their relative depth to the camera, ensuring correct
    /// occlusion and object visibility.
    ///
    /// The depth texture is created with the `Depth32Float` format, which
    /// provides high precision for depth calculations. It is resized as
    /// needed when the window or surface size changes.
    depth_texture_view: wgpu::TextureView,

    /// A renderer instance for rendering GUI elements created with `egui`.
    ///
    /// This component is responsible for translating `egui`'s GUI
    /// widgets and visuals into textures and commands that can be
    /// rendered on the GPU using `wgpu`. It manages the lifecycle of
    /// resources such as textures and buffers required for GUI rendering.
    ///
    /// The `egui_renderer` is tightly integrated with the rest of the
    /// `Renderer`, ensuring seamless rendering alongside 3D scene
    /// elements. It processes `egui` paint jobs and textures, and
    /// combines them into the overall frame rendering pipeline.
    ///
    /// This field is initialized during the creation of the `Renderer`
    /// struct and updated dynamically to reflect GUI state changes.
    egui_renderer: egui_wgpu::Renderer,

    /// The application's 3D scene, responsible for managing objects, transformations,
    /// lighting, and other scene-specific logic.
    ///
    /// This field encapsulates the data and behavior needed to render and update the 3D scene.
    /// It includes resources such as vertex and index buffers, shader pipelines, and
    /// transformation matrices, ensuring that objects within the scene are rendered correctly.
    ///
    /// The `Scene` struct also handles per-frame updates, such as applying animations,
    /// updating object positions, and recalculating camera matrices. It interacts closely with
    /// the GPU to ensure efficient rendering.
    ///
    /// The scene is updated during the rendering process (`render_frame`), where it processes
    /// changes in the application's state or interactions initiated by the user.
    scene: Scene,
}

/// Implementation of the `Renderer` struct, which provides methods for managing
/// and handling rendering tasks related to both the GUI and the 3D scene.
///
/// The `Renderer` struct integrates WGPU-based rendering for 3D graphics
/// and the `egui` framework for GUI rendering, combining them into
/// an efficient and reusable rendering pipeline.
///
/// # Constants
///
/// - `DEPTH_FORMAT`: Defines the texture format used for the depth buffer,
///   which is critical for ensuring proper depth calculations in 3D rendering.
///
/// # Methods
///
/// - `new`: Asynchronous function to initialize the `Renderer` with specified
///   GPU resources, depth textures, an `egui` renderer, and a 3D scene.
///   This function prepares all essential components for rendering and is
///   typically called during application startup.
///
/// - `resize`: Adjusts the size of GPU resources, such as the depth texture,
///   to match the new dimensions of the window or surface. This is critical
///   to maintaining proper rendering behavior when the window size changes.
///
/// - `render_frame`: Executes the rendering of a single frame, which includes:
///   - Updating the 3D scene based on the elapsed time.
///   - Processing `egui`-related GUI commands, such as texture updates.
///   - Issuing rendering commands to the GPU, combining GUI and 3D scene visuals,
///     and presenting the final output to the surface.
///
/// The `Renderer` implementation is designed to be used in an application's main loop,
/// updating and rendering the scene and GUI in real-time.
impl Renderer {
    /// The texture format used for the depth buffer in 3D rendering.
    ///
    /// This constant defines the format of the depth buffer as `Depth32Float`.
    /// It provides high precision for depth calculations, which is critical for
    /// rendering 3D scenes with accurate occlusion and depth-testing behavior.
    ///
    /// The depth buffer ensures that objects closer to the camera are drawn
    /// on top of those farther away, contributing to the realism of the scene.
    /// This format is particularly effective for applications that require precise
    /// depth calculations, such as rendering large, complex 3D environments.
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    /// Creates a new instance of the `Renderer` struct, initializing all required components.
    ///
    /// # Parameters
    ///
    /// - `window`: A target that represents the window or surface to which the application renders.
    /// - `width`: The width of the rendering surface in logical pixels.
    /// - `height`: The height of the rendering surface in logical pixels.
    ///
    /// # Returns
    ///
    /// A new `Renderer` instance with initialized GPU resources, depth textures,
    /// an `egui` renderer, and a 3D scene. This setup provides the necessary components
    /// for rendering both GUI elements and the 3D scene.
    ///
    /// # Asynchronous Behavior
    ///
    /// The function performs asynchronous operations to initialize GPU resources.
    /// It is expected to be called during the application’s startup, often within an
    /// asynchronous runtime or context.
    ///
    /// # Example
    ///
    /// ```rust
    /// let renderer = Renderer::new(window, 800, 600).await;
    /// ```
    ///
    /// In this example, a `Renderer` is initialized with a window surface and initial dimensions.
    ///
    /// # Panics
    ///
    /// This function may panic if GPU resources or the rendering environment cannot
    /// be properly initialized.
    pub async fn new(
        window: impl Into<wgpu::SurfaceTarget<'static>>,
        width: u32,
        height: u32,
    ) -> Self {
        // The GPU instance, responsible for managing the device, queue, and other
        // rendering-related resources required for interacting with the graphics hardware.
        //
        // This instance is initialized asynchronously and handles:
        // - Configuring the surface for rendering with appropriate settings.
        // - Managing the GPU device and command queue for issuing rendering commands.
        // - Creating and maintaining textures, buffers, and pipelines utilized during rendering.
        //
        // The `Gpu` struct is a key component for all rendering operations, abstracting
        // low-level GPU interactions and providing higher-level methods for resource creation
        // and management.
        let gpu = Gpu::new_async(window, width, height).await;

        // The texture view for the depth buffer used during 3D rendering.
        //
        // This texture view represents a depth buffer that is used to store depth
        // information for rendering operations. The depth buffer ensures correct
        // occlusion in 3D scenes, allowing closer objects to appear in front of those
        // farther away.
        //
        // The texture view is configured to match the dimensions of the rendering
        // surface and uses the depth format specified by `DEPTH_FORMAT`. It is
        // recreated whenever the rendering surface is resized to ensure accurate
        // depth calculations.
        //
        // This resource is critical for 3D rendering and is used as part of the
        // pipeline configuration.
        let depth_texture_view = gpu.create_depth_texture(width, height);

        // The `egui_renderer` is used to render the GUI elements within the application.
        //
        // This renderer integrates the `egui` framework with the GPU, enabling the
        // efficient rendering of GUI primitives on top of the 3D scene. It serves as
        // an important component of the rendering pipeline, ensuring that user-facing
        // interface elements are visually updated and properly displayed.
        //
        // # Details
        //
        // - Uses the `egui_wgpu` crate to bridge the `egui` GUI library and the `wgpu` graphics API.
        // - Configured with the device, surface format, and depth format to ensure compatibility
        //   with the GPU resources and rendering setup.
        // - Supports rendering of transparent and layered GUI elements, utilizing the depth buffer
        //   when required.
        //
        // This renderer is essential for applications with graphical interfaces, providing a bridge
        // between the interactive GUI and the underlying rendering engine.
        let egui_renderer = egui_wgpu::Renderer::new(
            &gpu.device,
            gpu.surface_config.format,
            Some(Self::DEPTH_FORMAT),
            1,
            false,
        );

        // The `scene` represents the 3D environment or visual content being rendered.
        //
        // This structure encapsulates all the objects, lighting, and other elements
        // that compose the 3D graphics rendered to the screen. It serves as the primary
        // container for managing and updating the visual state of the application.
        //
        // # Details
        //
        // - Configured using the GPU device and surface format to ensure compatibility
        //   with GPU resources and rendering workflows.
        // - Manages 3D models, textures, shaders, and other scene-related data.
        // - Integrates with the rendering pipeline, providing a primary source of data
        //   for rendering operations.
        //
        // The `scene` is updated and rendered as part of the rendering loop, reacting
        // to user input, animations, or external state to create an interactive and
        // dynamic 3D experience.
        let scene = Scene::new(&gpu.device, gpu.surface_format);

        Self {
            gpu,
            depth_texture_view,
            egui_renderer,
            scene,
        }
    }

    /// Resizes the rendering components to match the new size of the window or rendering surface.
    ///
    /// # Parameters
    ///
    /// - `width`: The new width of the rendering surface in logical pixels.
    /// - `height`: The new height of the rendering surface in logical pixels.
    ///
    /// # Details
    ///
    /// This method adjusts GPU resources, including the depth texture, to match
    /// the updated dimensions of the application window or surface. It ensures
    /// proper rendering by keeping the depth buffer and other components in sync
    /// with the current surface size.
    ///
    /// This function is typically called within the application when the user
    /// resizes the window, ensuring that the renderer properly updates to fit
    /// the new size.
    ///
    /// # Example
    ///
    /// ```rust
    /// renderer.resize(new_width, new_height);
    /// ```
    ///
    /// In this example, the renderer resizes its internal GPU resources to accommodate
    /// the updated surface dimensions.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.gpu.resize(width, height);
        self.depth_texture_view = self.gpu.create_depth_texture(width, height);
    }

    /// Renders a single frame, combining 3D scene rendering and `egui` GUI rendering.
    ///
    /// # Parameters
    ///
    /// - `screen_descriptor`: Describes the screen properties, such as size and DPI,
    ///   which are required by the `egui` renderer to render correctly.
    /// - `paint_jobs`: A collection of `egui` primitives (draw instructions)
    ///   to be rendered on the screen.
    /// - `textures_delta`: Contains information about added or removed textures
    ///   that are used by `egui` for rendering.
    /// - `delta_time`: The time elapsed since the last frame, which is used
    ///   for animations and time-dependent logic in the 3D scene.
    ///
    /// # Details
    ///
    /// This function coordinates the rendering of the GUI and the 3D scene. It updates
    /// the scene based on the elapsed time, updates `egui` textures, and manages the
    /// GPU command encoder for rendering to the surface. It performs the following steps:
    ///
    /// 1. Updates the 3D scene's state based on the `delta_time` to account for animations
    ///    or any other time-dependent behavior.
    /// 2. Updates and synchronizes `egui` textures (loading or freeing them as needed).
    /// 3. Executes `egui`'s paint jobs and integrates them with the rendered output.
    /// 4. Renders the frame to the surface, using a render pass that includes a color attachment
    ///    for the main scene and a depth attachment for proper depth-testing.
    ///
    /// # Panics
    ///
    /// This function will panic if the process of retrieving the surface texture fails. This
    /// can occur if the surface becomes invalid (e.g., if the window is resized or destroyed).
    ///
    /// # Example
    ///
    /// ```rust
    /// renderer.render_frame(
    ///     screen_descriptor,
    ///     paint_jobs,
    ///     textures_delta,
    ///     delta_time,
    /// );
    /// ```
    ///
    /// In this example, the function is called to render one frame, using the provided
    /// `egui` render data, screen dimensions, and the time elapsed since the previous frame.
    pub fn render_frame(
        &mut self,
        screen_descriptor: egui_wgpu::ScreenDescriptor,
        paint_jobs: Vec<egui::epaint::ClippedPrimitive>,
        textures_delta: egui::TexturesDelta,
        delta_time: crate::Duration,
    ) {
        // The elapsed time since the last frame, in seconds, represented as a 32-bit floating-point number.
        // This variable is used to update the state of the 3D scene, animations, and other
        // time-dependent logic within the render pipeline.
        //
        // `delta_time` facilitates smooth animations and transitions by allowing computations
        // to take the elapsed time into account, ensuring consistent behavior regardless of frame rate.
        let delta_time = delta_time.as_secs_f32();

        self.scene
            .update(&self.gpu.queue, self.gpu.aspect_ratio(), delta_time);

        // This loop iterates over all texture changes in the `textures_delta.set` map,
        // where `id` is the unique identifier for a texture and `image_delta` describes
        // the changes to be applied to that texture. For each entry, it updates
        // the corresponding texture in the `egui_renderer` using the provided
        // `gpu.device` and `gpu.queue`. This ensures that `egui` textures are
        // synchronized with changes made to them.
        for (id, image_delta) in &textures_delta.set {
            self.egui_renderer
                .update_texture(&self.gpu.device, &self.gpu.queue, *id, image_delta);
        }

        // Iterate through all texture updates specified in the `textures_delta.set` map.
        // Each entry in the map consists of a texture `id` (a unique identifier) and
        // an `image_delta` (describing how the texture should be updated). The `egui_renderer`
        // is then instructed to update these textures using the provided `gpu.device` and
        // `gpu.queue` for GPU operations, ensuring `egui` texture data is synchronized with
        // the latest changes.
        for id in &textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        // A command encoder used to record a series of GPU commands for execution.
        //
        // The `encoder` is responsible for managing the commands that will be sent to the GPU.
        // It is used to record operations such as rendering commands, buffer updates, and texture
        // manipulations. After recording, the commands are submitted for execution on the GPU queue.
        //
        // # Details
        //
        // - The encoder is created using the `wgpu::Device` method `create_command_encoder`.
        // - It has a descriptive label associated with it (`"Render Encoder"`) for easier debugging and profiling.
        // - This encoder is used to manage commands for updating `egui` buffers, rendering the 3D scene,
        //   updating textures, and executing draw calls on the primary surface texture.
        //
        // At the end of the rendering process, the recorded commands in the encoder are finalized by
        // calling the `.finish()` method and submitted to the GPU queue for execution.
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        self.egui_renderer.update_buffers(
            &self.gpu.device,
            &self.gpu.queue,
            &mut encoder,
            &paint_jobs,
            &screen_descriptor,
        );

        // Represents the texture for the current frame, retrieved from the surface.
        //
        // The `surface_texture` is obtained using the `get_current_texture()` method on the GPU surface.
        // It represents the texture that will be used as the rendering target for the current frame.
        // Rendering commands are issued to draw content onto this texture.
        //
        // # Details
        //
        // - This texture is tied to the window or canvas the application is rendering to.
        // - The texture is presented using the `present()` method once rendering has finished.
        // - If acquiring the texture fails (e.g., due to a lost surface), an error will be raised.
        //
        // The `surface_texture` is vital for ensuring rendered frames are output to the display.
        let surface_texture = self
            .gpu
            .surface
            .get_current_texture()
            .expect("Failed to get surface texture!");

        // Represents a view of the texture for the current frame.
        //
        // The `surface_texture_view` is created from the `surface_texture` using the `create_view` method.
        // It defines how the texture resource should be accessed and what properties it should expose
        // for rendering operations. This is especially useful when multiple views of the same texture
        // are required with varying configurations.
        //
        // # Details
        //
        // - The view is created with the `wgpu::TextureViewDescriptor`, which specifies configuration
        //   options such as format, dimension, aspect, and mip levels.
        // - The `aspect` defaults to full visibility (color only for color textures).
        // - The view is required as input for render operations like `RenderPass` or `RenderBundle`.
        // - This view allows the GPU to interpret the `surface_texture` data correctly when rendering
        //   the frame onto the screen.
        let surface_texture_view =
            surface_texture
                .texture
                .create_view(&wgpu::TextureViewDescriptor {
                    label: wgpu::Label::default(),
                    aspect: wgpu::TextureAspect::default(),
                    format: Some(self.gpu.surface_format),
                    dimension: None,
                    base_mip_level: 0,
                    mip_level_count: None,
                    base_array_layer: 0,
                    array_layer_count: None,
                    usage: None,
                });

        encoder.insert_debug_marker("Render scene");

        // This scope around the crate::render_pass prevents the
        // crate::render_pass from holding a borrow to the encoder,
        // which would prevent calling `.finish()` in
        // preparation for queue submission.
        {
            // Represents the render pass for issuing rendering commands to the GPU.
            //
            // The `render_pass` variable is used to manage and record a sequence of rendering operations,
            // such as setting pipeline states, drawing, and clearing attachments. This ensures the GPU
            // can execute these operations efficiently.
            //
            // # Details
            //
            // - A `render_pass` is created using the `encoder.begin_render_pass()` method with a
            //   `wgpu::RenderPassDescriptor` that specifies configurations for color and depth attachments.
            // - The `color_attachments` field associates the render target texture (or view) with the
            //   operations to be performed (e.g., clearing or storing color data).
            // - The `depth_stencil_attachment` field manages the depth and stencil buffers, ensuring proper
            //   depth testing and rendering order.
            //
            // This render pass is scoped to ensure the `'encode` lifetime of the encoder is not held after
            // the render pass completes, allowing the `encoder` to be finalized with the `.finish()` method.
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.19,
                            g: 0.24,
                            b: 0.42,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            self.scene.render(&mut render_pass);

            self.egui_renderer.render(
                &mut render_pass.forget_lifetime(),
                &paint_jobs,
                &screen_descriptor,
            );
        }

        self.gpu.queue.submit(std::iter::once(encoder.finish()));
        surface_texture.present();
    }
}
