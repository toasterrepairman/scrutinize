use gtk::prelude::*;
use gtk::{glib, Box as GtkBox, Orientation, ScrolledWindow, DrawingArea};
use adw::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

use crate::gguf_parser::{GGUFFile, TensorInfo};

#[derive(Clone)]
pub struct TensorPage {
    widget: adw::Bin,
    tensor_list: gtk::ListView,
    memory_viz: DrawingArea,
    tensors: Rc<RefCell<Vec<TensorInfo>>>,
    gguf_file: Rc<RefCell<Option<GGUFFile>>>, // Store reference to GGUFFile for optimized access
    system_info: Rc<RefCell<SystemMemoryInfo>>,
    file_path: Rc<RefCell<Option<std::path::PathBuf>>>,
}

/// Calculate adaptive maximum elements based on tensor size and display constraints
/// Implements intelligent scaling to preserve structure while maintaining performance
fn calculate_adaptive_max_elements(total_elements: usize) -> usize {
    // Base limits for different size categories - more conservative to prevent crashes
    match total_elements {
        // Small tensors: show everything, no downsampling needed
        0..=25_000 => total_elements,

        // Medium tensors: allow up to 500K elements for good detail
        25_001..=500_000 => {
            // Calculate reasonable limit that preserves detail but fits in widget
            std::cmp::min(total_elements, 500_000)
        }

        // Large tensors: smart downsampling with reasonable limits
        500_001..=5_000_000 => {
            // Allow up to 1M elements with smart downsampling
            std::cmp::min(total_elements, 1_000_000)
        }

        // Very large tensors: aggressive downsampling to fit viewport
        5_000_001..=50_000_000 => {
            // Target ~1K x 1K resolution max = 1M elements
            1_000_000
        }

        // Extremely large tensors: strict limits to prevent crashes
        _ => {
            // Cap at 1.5M elements maximum (fits in ~1200x1200 viewport)
            1_500_000
        }
    }
}

#[derive(Clone, Debug)]
struct SystemMemoryInfo {
    system_ram_bytes: u64,
    system_ram_used_bytes: u64,
    gpus: Vec<GPUInfo>,
}

#[derive(Clone, Debug)]
struct GPUInfo {
    name: String,
    vram_bytes: u64,
    vram_used_bytes: u64,
}

impl TensorPage {
    pub fn new() -> Self {
        let main_box = GtkBox::new(Orientation::Vertical, 0);

        // Memory visualization section
        let memory_group = adw::PreferencesGroup::builder()
            .title("Memory Requirements")
            .description("Visualization of how the model fits in your system")
            .margin_top(12)
            .margin_bottom(12)
            .margin_start(12)
            .margin_end(12)
            .build();

        let memory_viz = DrawingArea::new();
        // Height will be set dynamically based on number of GPUs
        memory_viz.set_hexpand(true);

        let viz_frame = gtk::Frame::builder()
            .child(&memory_viz)
            .build();

        memory_group.add(&viz_frame);
        main_box.append(&memory_group);

        // Tensor list
        let tensor_group = adw::PreferencesGroup::builder()
            .title("Tensor Details")
            .margin_top(12)
            .margin_bottom(12)
            .margin_start(12)
            .margin_end(12)
            .build();

        let tensors = Rc::new(RefCell::new(Vec::new()));
        let gguf_file = Rc::new(RefCell::new(None));
        let file_path: Rc<RefCell<Option<std::path::PathBuf>>> = Rc::new(RefCell::new(None));

        let model = gio::ListStore::new::<glib::BoxedAnyObject>();
        let selection_model = gtk::NoSelection::new(Some(model.clone()));

        let factory = gtk::SignalListItemFactory::new();
        let file_path_for_setup = Rc::clone(&file_path);

        factory.connect_setup(move |_, list_item| {
            let row = adw::ActionRow::builder()
                .activatable(true)
                .build();

            let type_label = gtk::Label::builder()
                .margin_end(8)
                .css_classes(["dim-label"])
                .build();
            row.add_suffix(&type_label);

            // Add click gesture
            let gesture = gtk::GestureClick::new();
            let file_path_weak = Rc::downgrade(&file_path_for_setup);
            gesture.connect_released(move |gesture, _, _, _| {
                if let Some(row) = gesture.widget().and_downcast_ref::<adw::ActionRow>() {
                    // Retrieve tensor data from object data
                    unsafe {
                        use glib::translate::ToGlibPtr;
                        let tensor_ptr = glib::gobject_ffi::g_object_get_data(
                            row.as_ptr() as *mut glib::gobject_ffi::GObject,
                            b"tensor-data\0".as_ptr() as *const i8,
                        );

                        if !tensor_ptr.is_null() {
                            let tensor = &*(tensor_ptr as *const TensorInfo);
                            if let Some(file_path_rc) = file_path_weak.upgrade() {
                                if let Some(ref path) = *file_path_rc.borrow() {
                                    TensorPage::show_tensor_popover(row.upcast_ref::<gtk::Widget>(), tensor, path);
                                }
                            }
                        }
                    }
                }
            });
            row.add_controller(gesture);

            let list_item = list_item.downcast_ref::<gtk::ListItem>().unwrap();
            list_item.set_child(Some(&row));
        });

        let file_path_for_bind = Rc::clone(&file_path);
        factory.connect_bind(move |_, list_item| {
            let list_item = list_item.downcast_ref::<gtk::ListItem>().unwrap();
            let item = list_item.item()
                .and_downcast::<glib::BoxedAnyObject>()
                .unwrap();
            let tensor = item.borrow::<TensorInfo>().clone();

            let row = list_item.child()
                .and_downcast::<adw::ActionRow>()
                .unwrap();

            row.set_title(&tensor.name);

            let subtitle = format!("Shape: {} • Size: {}",
                tensor.shape_string(),
                format_bytes(tensor.size_bytes));
            row.set_subtitle(&subtitle);

            // Update type label - find the suffix label
            let mut suffix_widgets = Vec::new();
            let mut current = row.first_child();
            while let Some(widget) = current {
                suffix_widgets.push(widget.clone());
                current = widget.next_sibling();
            }

            // The type label should be a suffix
            for widget in suffix_widgets {
                if let Ok(label) = widget.downcast::<gtk::Label>() {
                    if label.css_classes().iter().any(|c| c == "dim-label") {
                        label.set_text(tensor.dtype.name());
                        break;
                    }
                }
            }

            // Store tensor data in the row using object data
            unsafe {
                use glib::translate::ToGlibPtr;
                let tensor_box = Box::new(tensor.clone());
                let tensor_ptr = Box::into_raw(tensor_box) as glib::ffi::gpointer;
                glib::gobject_ffi::g_object_set_data_full(
                    row.as_ptr() as *mut glib::gobject_ffi::GObject,
                    b"tensor-data\0".as_ptr() as *const i8,
                    tensor_ptr,
                    Some(std::mem::transmute::<_, unsafe extern "C" fn(glib::ffi::gpointer)>(
                        free_tensor_data as *const ()
                    )),
                );
            }
        });

        let tensor_list = gtk::ListView::new(Some(selection_model.clone()), Some(factory));

        let scrolled = ScrolledWindow::builder()
            .hscrollbar_policy(gtk::PolicyType::Never)
            .vscrollbar_policy(gtk::PolicyType::Automatic)
            .child(&tensor_list)
            .height_request(300)
            .vexpand(true)
            .build();

        tensor_group.add(&scrolled);
        main_box.append(&tensor_group);

        let widget = adw::Bin::new();
        widget.set_child(Some(&main_box));

        // Get system memory info
        let system_info = Rc::new(RefCell::new(get_system_memory_info()));

        let page = Self {
            widget,
            tensor_list,
            memory_viz: memory_viz.clone(),
            tensors: tensors.clone(),
            gguf_file: gguf_file.clone(),
            system_info: system_info.clone(),
            file_path: file_path.clone(),
        };

        // Connect drawing
        let tensors_weak = Rc::downgrade(&tensors);
        let system_info_weak = Rc::downgrade(&system_info);
        memory_viz.set_draw_func(move |_, cr, width, height| {
            if let (Some(tensors), Some(sys_info)) = (tensors_weak.upgrade(), system_info_weak.upgrade()) {
                draw_memory_visualization(cr, width, height, &tensors.borrow(), &sys_info.borrow());
            }
        });

        page
    }

    pub fn widget(&self) -> &adw::Bin {
        &self.widget
    }

    pub fn load_tensors(&self, gguf_file: &GGUFFile) {
        *self.tensors.borrow_mut() = gguf_file.tensors.clone();
        *self.gguf_file.borrow_mut() = Some(gguf_file.clone()); // Store GGUFFile for optimized access

        // Update list view
        if let Some(selection_model) = self.tensor_list.model() {
            if let Some(selection_model) = selection_model.downcast_ref::<gtk::NoSelection>() {
                if let Some(model) = selection_model.model() {
                    let list_store = model.downcast_ref::<gio::ListStore>().unwrap();
                    list_store.remove_all();

                    for tensor in gguf_file.tensors.iter() {
                        list_store.append(&glib::BoxedAnyObject::new(tensor.clone()));
                    }
                }
            }
        }

        // Calculate and set dynamic height based on system configuration
        let height = calculate_visualization_height(&self.system_info.borrow());
        self.memory_viz.set_content_height(height);

        // Trigger redraw of memory visualization
        self.memory_viz.queue_draw();
    }

    pub fn set_file_path(&self, path: std::path::PathBuf) {
        *self.file_path.borrow_mut() = Some(path);
    }

    /// Optimized tensor data reading using memory mapping when available
    /// Falls back to file I/O if memory mapping is not initialized
    async fn read_tensor_data_optimized(
        &self,
        tensor: &TensorInfo,
        max_elements: usize,
        slice_selection: Option<&crate::tensor_slice::SliceSelection>
    ) -> Result<Vec<f32>, String> {
        // Try to use optimized memory-mapped reading first
        if let Some(ref gguf_file) = *self.gguf_file.borrow() {
            // Check if memory mapping is available
            if let Ok(mmap) = gguf_file.get_mmap() {
                // Use memory-mapped reading (much faster)
                return tensor.read_tensor_data_optimized(&*mmap, max_elements, slice_selection);
            }
        }

        // Fall back to file I/O if memory mapping is not available
        if let Some(ref path) = *self.file_path.borrow() {
            return tensor.read_tensor_data_with_slice(path, max_elements, slice_selection);
        }

        Err("No file path available for tensor reading".to_string())
    }

    fn show_tensor_popover(parent: &gtk::Widget, tensor: &TensorInfo, file_path: &std::path::Path) {
        use crate::heatmap_widget::HeatmapWidget;

        // Ensure parent is realized before creating popover
        if !parent.is_realized() {
            parent.realize();
        }

        let popover = gtk::Popover::new();
        popover.set_parent(parent);
        popover.set_position(gtk::PositionType::Bottom);
        popover.set_width_request(450);
        popover.set_autohide(true);
        popover.set_has_arrow(true);

        let content = GtkBox::new(Orientation::Vertical, 12);
        content.set_margin_top(12);
        content.set_margin_bottom(12);
        content.set_margin_start(12);
        content.set_margin_end(12);

        // Title
        let title = gtk::Label::builder()
            .label(&format!("<b>{}</b>", glib::markup_escape_text(&tensor.name)))
            .use_markup(true)
            .halign(gtk::Align::Start)
            .wrap(true)
            .wrap_mode(gtk::pango::WrapMode::WordChar)
            .build();
        content.append(&title);

        // Metadata grid
        let meta_grid = gtk::Grid::builder()
            .column_spacing(12)
            .row_spacing(6)
            .build();

        let labels = [
            ("Shape:", tensor.shape_string()),
            ("Type:", tensor.dtype.name().to_string()),
            ("Elements:", format!("{}", tensor.element_count())),
            ("Size:", format_bytes(tensor.size_bytes)),
            ("Offset:", format!("0x{:X}", tensor.offset)),
        ];

        for (row, (key, value)) in labels.iter().enumerate() {
            let key_label = gtk::Label::builder()
                .label(*key)
                .halign(gtk::Align::End)
                .css_classes(vec!["dim-label".to_string()])
                .build();
            let value_label = gtk::Label::builder()
                .label(value)
                .halign(gtk::Align::Start)
                .selectable(true)
                .build();

            meta_grid.attach(&key_label, 0, row as i32, 1, 1);
            meta_grid.attach(&value_label, 1, row as i32, 1, 1);
        }

        content.append(&meta_grid);

        // Separator
        content.append(&gtk::Separator::new(Orientation::Horizontal));

        // Visualization header with toggle button
        let viz_header = GtkBox::new(Orientation::Horizontal, 8);
        let viz_label = gtk::Label::builder()
            .label("<b>Tensor Visualization</b>")
            .use_markup(true)
            .halign(gtk::Align::Start)
            .hexpand(true)
            .build();
        viz_header.append(&viz_label);

        // Toggle button for heatmap/histogram
        let toggle_button = gtk::Button::builder()
            .label("Histogram")
            .halign(gtk::Align::End)
            .css_classes(vec!["flat".to_string()])
            .sensitive(false) // Disabled until data loads
            .build();
        viz_header.append(&toggle_button);

        content.append(&viz_header);

        // Loading indicator with spinner
        let loading_box = GtkBox::new(Orientation::Horizontal, 8);
        loading_box.set_halign(gtk::Align::Center);
        loading_box.set_valign(gtk::Align::Center);
        loading_box.set_margin_top(12);
        loading_box.set_margin_bottom(12);

        let spinner = gtk::Spinner::builder()
            .spinning(true)
            .build();
        loading_box.append(&spinner);

        let status_label = gtk::Label::builder()
            .label("Loading tensor data...")
            .css_classes(vec!["dim-label".to_string()])
            .build();
        loading_box.append(&status_label);

        content.append(&loading_box);

        let heatmap_widget = HeatmapWidget::new();
        heatmap_widget.widget().set_visible(false);
        content.append(heatmap_widget.widget());

        // Statistics section (initially hidden)
        let stats_separator = gtk::Separator::new(Orientation::Horizontal);
        stats_separator.set_visible(false);
        content.append(&stats_separator);

        let stats_label = gtk::Label::builder()
            .label("<b>Statistics</b>")
            .use_markup(true)
            .halign(gtk::Align::Start)
            .visible(false)
            .build();
        content.append(&stats_label);

        let stats_grid = gtk::Grid::builder()
            .column_spacing(12)
            .row_spacing(6)
            .visible(false)
            .build();
        content.append(&stats_grid);

        popover.set_child(Some(&content));

        // Clean up popover when it's closed
        popover.connect_closed(|popover| {
            popover.unparent();
        });

        popover.popup();

        // Load tensor data asynchronously to avoid blocking UI
        let tensor_clone = tensor.clone();
        let file_path_clone = file_path.to_path_buf();
        let heatmap_weak = heatmap_widget.clone();
        let loading_box_weak = loading_box.downgrade();
        let status_weak = status_label.downgrade();
        let stats_separator_weak = stats_separator.downgrade();
        let stats_label_weak = stats_label.downgrade();
        let stats_grid_weak = stats_grid.downgrade();
        let toggle_btn_weak = toggle_button.downgrade();

        // Use async-std for true async file I/O
        glib::spawn_future_local(async move {
            use crate::tensor_slice::SliceSelection;

            // Calculate adaptive max elements based on tensor dimensions
            // Allow larger tensors but implement intelligent downsampling
            let total_elements = tensor_clone.dimensions.iter().product::<u64>() as usize;
            let max_elements = calculate_adaptive_max_elements(total_elements);

            eprintln!("Tensor {} loaded: shape={:?}, total_elements={}, dtype={:?}, max_elements={}",
                tensor_clone.name, tensor_clone.dimensions, total_elements, tensor_clone.dtype, max_elements);

            // Create smart slice selection for 3D+ tensors
            let slice_selection = if tensor_clone.dimensions.len() > 2 {
                let sel = SliceSelection::smart_default(&tensor_clone.name, tensor_clone.dimensions.clone());
                eprintln!("Created slice selection for {}: shape={:?}, needs_slicing={}",
                    tensor_clone.name, tensor_clone.dimensions, sel.needs_slicing());
                Some(sel)
            } else {
                eprintln!("Tensor {} is 2D or less (shape={:?}), no slice selection needed",
                    tensor_clone.name, tensor_clone.dimensions);
                None
            };

            // Set slice selection in heatmap widget (this shows the controls)
            eprintln!("Setting slice selection on heatmap widget...");
            heatmap_weak.set_slice_selection(slice_selection.clone());
            eprintln!("Slice selection set successfully");

            // Set up callback to reload data when slice changes
            let tensor_for_callback = tensor_clone.clone();
            let file_path_for_callback = file_path_clone.clone();
            let heatmap_for_callback = heatmap_weak.clone();
            let status_for_callback = status_weak.clone();
            let loading_for_callback = loading_box_weak.clone();

            heatmap_weak.set_on_slice_change(move |new_selection| {
                eprintln!("Slice changed, reloading data...");
                let tensor = tensor_for_callback.clone();
                let path = file_path_for_callback.clone();
                let heatmap = heatmap_for_callback.clone();
                let status = status_for_callback.clone();
                let loading = loading_for_callback.clone();

                // Show loading indicator
                if let Some(loading_box) = loading.upgrade() {
                    loading_box.set_visible(true);
                }
                if let Some(label) = status.upgrade() {
                    label.set_label("Loading slice...");
                    label.remove_css_class("error");
                }

                // Spawn async task to reload data
                glib::spawn_future_local(async move {
                    // Calculate adaptive max elements for the new slice
                    let slice_elements = if let (h, w) = new_selection.slice_shape() {
                        (h * w) as usize
                    } else {
                        tensor.element_count() as usize
                    };
                    let max_elements = calculate_adaptive_max_elements(slice_elements);

                    // Run blocking I/O in a background thread to avoid blocking UI
                    let (tx, rx) = async_channel::unbounded();
                    let tensor = tensor.clone();
                    let path = path.clone();
                    let new_selection_for_thread = new_selection.clone();

                    std::thread::spawn(move || {
                        let result = tensor.read_tensor_data_with_slice(&path, max_elements, Some(&new_selection_for_thread));
                        let _ = tx.send_blocking(result);
                    });

                    let result = rx.recv().await.unwrap_or_else(|_| Err("Thread panicked".to_string()));

                    match result {
                        Ok(data) => {
                            let (h, w) = new_selection.slice_shape();
                            heatmap.set_data(data, &vec![h, w]);
                            eprintln!("Data reloaded successfully for new slice");

                            // Hide loading indicator
                            if let Some(loading_box) = loading.upgrade() {
                                loading_box.set_visible(false);
                            }
                        }
                        Err(e) => {
                            eprintln!("Error reloading slice data: {}", e);

                            // Hide spinner, show error
                            if let Some(loading_box) = loading.upgrade() {
                                loading_box.set_visible(false);
                            }
                            if let Some(label) = status.upgrade() {
                                label.set_label(&format!("Error: {}", e));
                                label.set_visible(true);
                                label.add_css_class("error");
                            }
                        }
                    }
                });
            });

            // Run initial data load in background thread to avoid blocking UI
            let (tx, rx) = async_channel::unbounded();
            let tensor = tensor_clone.clone();
            let path = file_path_clone.clone();
            let sel = slice_selection.clone();

            std::thread::spawn(move || {
                let result = tensor.read_tensor_data_with_slice(&path, max_elements, sel.as_ref());
                let _ = tx.send_blocking(result);
            });

            let result = rx.recv().await.unwrap_or_else(|_| Err("Thread panicked".to_string()));

            match result {
                Ok(data) => {
                    // Get the shape to display based on slice selection
                    let display_shape = if let Some(ref sel) = slice_selection {
                        let (h, w) = sel.slice_shape();
                        vec![h, w]
                    } else {
                        tensor_clone.dimensions.clone()
                    };

                    // Auto-detect 1D tensors and switch to line plot mode
                    let is_1d = tensor_clone.dimensions.len() == 1;
                    if is_1d {
                        heatmap_weak.set_display_mode_line_plot();
                        eprintln!("Tensor {} is 1D, using line plot mode", tensor_clone.name);
                    }

                    heatmap_weak.set_data(data, &display_shape);
                    heatmap_weak.widget().set_visible(true);

                    // Hide loading indicator
                    if let Some(loading_box) = loading_box_weak.upgrade() {
                        loading_box.set_visible(false);
                    }

                    // Enable the toggle button now that data is loaded
                    if let Some(btn) = toggle_btn_weak.upgrade() {
                        btn.set_sensitive(true);

                        // Set initial button label based on tensor dimensionality
                        if is_1d {
                            btn.set_label("Histogram"); // 1D starts in Line Plot, so button goes to Histogram
                        } else {
                            btn.set_label("Histogram"); // 2D+ starts in Heatmap, so button goes to Histogram
                        }

                        // Connect click handler to toggle display mode
                        let heatmap_for_toggle = heatmap_weak.clone();
                        btn.connect_clicked(move |button| {
                            heatmap_for_toggle.toggle_display_mode();
                            // Update button label based on current mode
                            let mode_label = heatmap_for_toggle.get_display_mode_label();
                            match mode_label.as_str() {
                                "Line Plot" => button.set_label("Histogram"),
                                "Histogram" => {
                                    // Determine next mode based on tensor dimensionality
                                    if heatmap_for_toggle.is_line_plot_mode() {
                                        // Was in line plot, go to histogram
                                        button.set_label("Heatmap");
                                    } else {
                                        // Was in heatmap or histogram, toggle between them
                                        button.set_label("Heatmap");
                                    }
                                }
                                "Heatmap" => button.set_label("Histogram"),
                                _ => button.set_label("Histogram"),
                            }
                        });
                    }
                }
                Err(e) => {
                    // Hide spinner, show error
                    if let Some(loading_box) = loading_box_weak.upgrade() {
                        loading_box.set_visible(false);
                    }
                    if let Some(label) = status_weak.upgrade() {
                        label.set_label(&format!("Error loading tensor: {}", e));
                        label.set_visible(true);
                        label.add_css_class("error");
                    }
                }
            }
        });
    }
}

fn calculate_visualization_height(system_info: &SystemMemoryInfo) -> i32 {
    let margin = 12.0;
    let bar_height = 52.0;
    let spacing = 8.0;
    let legend_height = 24.0; // Space for legend at bottom

    // Calculate number of memory regions (GPUs + System RAM)
    let num_regions = system_info.gpus.len() + 1; // +1 for System RAM

    // Calculate total height needed
    let total_height = margin // top margin
        + (bar_height * num_regions as f64) // bars for each region
        + (spacing * num_regions as f64) // spacing after each bar (including last one for legend)
        + legend_height; // space for legend

    total_height.ceil() as i32
}

fn get_system_memory_info() -> SystemMemoryInfo {
    let (system_ram_bytes, system_ram_used_bytes) = get_system_ram_info();
    let gpus = get_gpu_info();

    SystemMemoryInfo {
        system_ram_bytes,
        system_ram_used_bytes,
        gpus,
    }
}

fn get_system_ram_info() -> (u64, u64) {
    // Try to read from /proc/meminfo on Linux
    if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
        let mut total_kb = 0u64;
        let mut available_kb = 0u64;

        for line in content.lines() {
            if line.starts_with("MemTotal:") {
                if let Some(kb_str) = line.split_whitespace().nth(1) {
                    total_kb = kb_str.parse::<u64>().unwrap_or(0);
                }
            } else if line.starts_with("MemAvailable:") {
                if let Some(kb_str) = line.split_whitespace().nth(1) {
                    available_kb = kb_str.parse::<u64>().unwrap_or(0);
                }
            }
        }

        if total_kb > 0 {
            let total_bytes = total_kb * 1024;
            let used_bytes = total_bytes.saturating_sub(available_kb * 1024);
            return (total_bytes, used_bytes);
        }
    }

    // Fallback: assume 16 GB total, 8 GB used
    (16 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024)
}

fn get_gpu_info() -> Vec<GPUInfo> {
    let mut gpus = Vec::new();

    // Try nvidia-smi for NVIDIA GPUs
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=name,memory.total,memory.used")
        .arg("--format=csv,noheader,nounits")
        .output() {
        if output.status.success() {
            if let Ok(output_str) = String::from_utf8(output.stdout) {
                for line in output_str.lines() {
                    let parts: Vec<&str> = line.split(',').collect();
                    if parts.len() >= 3 {
                        let name = parts[0].trim().to_string();
                        if let (Ok(vram_mb), Ok(used_mb)) = (
                            parts[1].trim().parse::<u64>(),
                            parts[2].trim().parse::<u64>()
                        ) {
                            gpus.push(GPUInfo {
                                name,
                                vram_bytes: vram_mb * 1024 * 1024,
                                vram_used_bytes: used_mb * 1024 * 1024,
                            });
                        }
                    }
                }
            }
        }
    }

    // Try rocm-smi for AMD GPUs
    if gpus.is_empty() {
        if let Ok(output) = std::process::Command::new("rocm-smi")
            .arg("--showmeminfo")
            .arg("vram")
            .output() {
            if output.status.success() {
                // AMD GPU detected, but parsing rocm-smi is complex
                // For now, add a placeholder
                gpus.push(GPUInfo {
                    name: "AMD GPU (detected)".to_string(),
                    vram_bytes: 8 * 1024 * 1024 * 1024, // Assume 8GB
                    vram_used_bytes: 0, // Unknown
                });
            }
        }
    }

    gpus
}

fn draw_memory_visualization(
    cr: &gtk::cairo::Context,
    width: i32,
    height: i32,
    tensors: &[TensorInfo],
    system_info: &SystemMemoryInfo,
) {
    if tensors.is_empty() {
        return;
    }

    let width = width as f64;
    let height = height as f64;
    let margin = 12.0;
    let bar_height = 52.0;
    let spacing = 8.0;
    let corner_radius = 6.0;

    // Calculate total model size
    let model_size: u64 = tensors.iter().map(|t| t.size_bytes).sum();

    // Calculate available memory (GPUs first, then system RAM)
    let mut memory_regions = Vec::new();
    let mut y_offset = margin;

    // Add GPU memory regions
    for (i, gpu) in system_info.gpus.iter().enumerate() {
        let display_name = truncate_device_name(&gpu.name, 30);
        memory_regions.push(MemoryRegion {
            name: if system_info.gpus.len() > 1 {
                format!("GPU {}: {}", i, display_name)
            } else {
                display_name
            },
            capacity: gpu.vram_bytes,
            used: gpu.vram_used_bytes,
            y_pos: y_offset,
            is_gpu: true,
        });
        y_offset += bar_height + spacing;
    }

    // Add system RAM
    memory_regions.push(MemoryRegion {
        name: "System RAM".to_string(),
        capacity: system_info.system_ram_bytes,
        used: system_info.system_ram_used_bytes,
        y_pos: y_offset,
        is_gpu: false,
    });

    // Calculate how much model fits in each region (accounting for already used memory)
    let mut remaining_model = model_size;
    let mut allocations = Vec::new();

    for region in &memory_regions {
        let available = region.capacity.saturating_sub(region.used);
        let allocated = remaining_model.min(available);
        allocations.push(allocated);
        remaining_model = remaining_model.saturating_sub(allocated);
    }

    let overflow = remaining_model > 0;

    // Setup font
    cr.select_font_face("Cantarell", gtk::cairo::FontSlant::Normal, gtk::cairo::FontWeight::Normal);
    cr.set_font_size(11.0);

    // Draw memory bars
    for (i, region) in memory_regions.iter().enumerate() {
        let model_allocated = allocations[i];
        let total_usage = region.used + model_allocated;
        let available_after = region.capacity.saturating_sub(region.used);

        let base_fill_ratio = region.used as f64 / region.capacity as f64;
        let model_fill_ratio = model_allocated as f64 / region.capacity as f64;
        let total_fill_ratio = total_usage as f64 / region.capacity as f64;

        // Use GNOME HIG colors (adapted for dark/light theme)
        // Background: card background
        cr.set_source_rgba(0.0, 0.0, 0.0, 0.1);
        draw_rounded_rect(cr, margin, region.y_pos, width - 2.0 * margin, bar_height, corner_radius);
        cr.fill().unwrap();

        let bar_margin = 8.0;
        let bar_inner_height = 20.0;
        let bar_y = region.y_pos + bar_height - bar_margin - bar_inner_height;
        let bar_width = width - 2.0 * margin - 2.0 * bar_margin;

        // Draw background bar (unfilled)
        cr.set_source_rgba(0.5, 0.5, 0.5, 0.2);
        draw_rounded_rect(cr, margin + bar_margin, bar_y, bar_width, bar_inner_height, 4.0);
        cr.fill().unwrap();

        // Draw currently used memory (darker)
        if region.used > 0 {
            let used_width = bar_width * base_fill_ratio;
            if region.is_gpu {
                cr.set_source_rgba(0.20, 0.47, 0.80, 0.6); // Blue (muted)
            } else {
                cr.set_source_rgba(0.37, 0.62, 0.31, 0.6); // Green (muted)
            }
            draw_rounded_rect(cr, margin + bar_margin, bar_y, used_width, bar_inner_height, 4.0);
            cr.fill().unwrap();
        }

        // Draw model allocation (brighter, overlaid)
        if model_allocated > 0 {
            let start_x = margin + bar_margin + (bar_width * base_fill_ratio);
            let model_width = bar_width * model_fill_ratio;

            // Determine color based on total usage after adding model
            if total_fill_ratio > 0.95 {
                cr.set_source_rgba(0.91, 0.28, 0.24, 1.0); // GNOME red (error)
            } else if total_fill_ratio > 0.85 {
                cr.set_source_rgba(0.95, 0.60, 0.00, 1.0); // GNOME orange (warning)
            } else if region.is_gpu {
                cr.set_source_rgba(0.25, 0.59, 0.95, 1.0); // GNOME blue (accent)
            } else {
                cr.set_source_rgba(0.46, 0.78, 0.39, 1.0); // GNOME green (success)
            }
            draw_rounded_rect(cr, start_x, bar_y, model_width, bar_inner_height, 4.0);
            cr.fill().unwrap();
        }

        // Draw text labels (responsive to container width)
        cr.set_source_rgba(1.0, 1.0, 1.0, 0.9);
        cr.set_font_size(10.5);

        // Determine available horizontal space for text
        let available_text_width = width - 2.0 * margin - 2.0 * bar_margin;

        // Device name (top left) with available memory
        let name_text = if model_allocated > 0 && available_after > model_allocated && available_text_width > 400.0 {
            format!("{} • {} available", region.name, format_bytes(available_after - model_allocated))
        } else {
            region.name.clone()
        };
        cr.move_to(margin + bar_margin, region.y_pos + 16.0);
        cr.show_text(&name_text).unwrap();

        // Memory stats (top right) - adapt based on available space
        cr.set_font_size(10.0);
        let stats_text = if available_text_width > 500.0 {
            // Full format with all details
            if model_allocated > 0 {
                format!("{} + {} / {} ({:.0}%)",
                    format_bytes(region.used),
                    format_bytes(model_allocated),
                    format_bytes(region.capacity),
                    total_fill_ratio * 100.0
                )
            } else {
                format!("{} / {} ({:.0}%)",
                    format_bytes(region.used),
                    format_bytes(region.capacity),
                    base_fill_ratio * 100.0
                )
            }
        } else if available_text_width > 300.0 {
            // Medium format - show consumed memory only
            if model_allocated > 0 {
                format!("{} + {}", format_bytes(region.used), format_bytes(model_allocated))
            } else {
                format!("{}", format_bytes(region.used))
            }
        } else {
            // Minimal format - show only LLM-consumed memory when very narrow
            if model_allocated > 0 {
                format_bytes(model_allocated)
            } else {
                format_bytes(region.used)
            }
        };

        let text_extents = cr.text_extents(&stats_text).unwrap();
        let stats_x = (width - margin - bar_margin - text_extents.width()).max(margin + bar_margin + 200.0);
        cr.move_to(stats_x, region.y_pos + 16.0);
        cr.set_source_rgba(1.0, 1.0, 1.0, 0.65);
        cr.show_text(&stats_text).unwrap();
    }

    // Draw overflow warning if needed
    if overflow {
        let warning_y = y_offset + bar_height + spacing;
        cr.set_source_rgba(0.91, 0.28, 0.24, 1.0); // GNOME red
        cr.set_font_size(11.0);
        cr.move_to(margin + 4.0, warning_y + 16.0);
        cr.show_text(&format!("⚠ Insufficient memory! {} additional space needed", format_bytes(remaining_model))).unwrap();
        y_offset = warning_y + 20.0;
    }

    // Draw compact legend at bottom
    let legend_y = y_offset + bar_height + spacing + 12.0;
    cr.set_font_size(9.5);

    // Model size label
    cr.set_source_rgba(1.0, 1.0, 1.0, 0.55);
    cr.move_to(margin + 4.0, legend_y);
    cr.show_text(&format!("Model: {}", format_bytes(model_size))).unwrap();

    // Legend items with color boxes - condensed spacing
    let legend_start_x = margin + 120.0;
    let box_size = 10.0;
    let box_y_offset = 7.0;
    let legend_item_spacing = 100.0; // Balanced spacing to prevent overlap

    // "Currently Used" legend
    cr.set_source_rgba(0.37, 0.62, 0.31, 0.6);
    draw_rounded_rect(cr, legend_start_x, legend_y - box_y_offset, box_size, box_size, 2.0);
    cr.fill().unwrap();

    cr.set_source_rgba(1.0, 1.0, 1.0, 0.75);
    cr.move_to(legend_start_x + box_size + 6.0, legend_y);
    cr.show_text("Currently Used").unwrap();

    // "Model" legend
    let model_x = legend_start_x + legend_item_spacing;
    cr.set_source_rgba(0.46, 0.78, 0.39, 1.0);
    draw_rounded_rect(cr, model_x, legend_y - box_y_offset, box_size, box_size, 2.0);
    cr.fill().unwrap();

    cr.set_source_rgba(1.0, 1.0, 1.0, 0.75);
    cr.move_to(model_x + box_size + 6.0, legend_y);
    cr.show_text("Model").unwrap();

    // "Warning" legend - only show when there's enough space
    if width > 450.0 {
        let warning_x = model_x + legend_item_spacing - 15.0; // Adjusted for tighter spacing
        cr.set_source_rgba(0.95, 0.60, 0.00, 1.0);
        draw_rounded_rect(cr, warning_x, legend_y - box_y_offset, box_size, box_size, 2.0);
        cr.fill().unwrap();

        cr.set_source_rgba(1.0, 1.0, 1.0, 0.75);
        cr.move_to(warning_x + box_size + 6.0, legend_y);
        cr.show_text(">85% Full").unwrap();
    }
}

fn draw_rounded_rect(cr: &gtk::cairo::Context, x: f64, y: f64, width: f64, height: f64, radius: f64) {
    let degrees = std::f64::consts::PI / 180.0;

    cr.new_sub_path();
    cr.arc(x + width - radius, y + radius, radius, -90.0 * degrees, 0.0 * degrees);
    cr.arc(x + width - radius, y + height - radius, radius, 0.0 * degrees, 90.0 * degrees);
    cr.arc(x + radius, y + height - radius, radius, 90.0 * degrees, 180.0 * degrees);
    cr.arc(x + radius, y + radius, radius, 180.0 * degrees, 270.0 * degrees);
    cr.close_path();
}

fn truncate_device_name(name: &str, max_len: usize) -> String {
    if name.len() <= max_len {
        return name.to_string();
    }

    // Try to intelligently truncate common GPU names
    let name = name
        .replace("NVIDIA ", "")
        .replace("GeForce ", "")
        .replace("AMD Radeon ", "")
        .replace("Radeon ", "");

    if name.len() <= max_len {
        return name;
    }

    // Truncate with ellipsis
    format!("{}…", &name[..max_len.saturating_sub(1)])
}

struct MemoryRegion {
    name: String,
    capacity: u64,
    used: u64,
    y_pos: f64,
    is_gpu: bool,
}

fn format_bytes(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = 1024 * KIB;
    const GIB: u64 = 1024 * MIB;

    if bytes >= GIB {
        format!("{:.2} GiB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.2} MiB", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{:.2} KiB", bytes as f64 / KIB as f64)
    } else {
        format!("{} B", bytes)
    }
}

// Callback for freeing tensor data stored in GObject data
unsafe extern "C" fn free_tensor_data(data: glib::ffi::gpointer) {
    if !data.is_null() {
        let _ = Box::from_raw(data as *mut TensorInfo);
    }
}
