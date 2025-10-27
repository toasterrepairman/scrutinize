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
    system_info: Rc<RefCell<SystemMemoryInfo>>,
    file_path: Rc<RefCell<Option<std::path::PathBuf>>>,
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

        // Loading label (will be replaced by heatmap or error)
        let status_label = gtk::Label::builder()
            .label("Loading tensor data...")
            .css_classes(vec!["dim-label".to_string()])
            .build();
        content.append(&status_label);

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
        let status_weak = status_label.downgrade();
        let stats_separator_weak = stats_separator.downgrade();
        let stats_label_weak = stats_label.downgrade();
        let stats_grid_weak = stats_grid.downgrade();
        let toggle_btn_weak = toggle_button.downgrade();

        glib::spawn_future_local(async move {
            use crate::tensor_slice::SliceSelection;

            // Determine max elements based on model size
            // For large models (>20GB), use more aggressive quantization
            let max_elements = if tensor_clone.size_bytes > 20 * 1024 * 1024 * 1024 {
                512 * 256  // 131k elements max for very large models
            } else if tensor_clone.size_bytes > 5 * 1024 * 1024 * 1024 {
                1024 * 512  // 524k elements for large models
            } else {
                2048 * 1024  // 2M elements for smaller models
            };

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
            heatmap_weak.set_on_slice_change(move |new_selection| {
                eprintln!("Slice changed, reloading data...");
                let tensor = tensor_for_callback.clone();
                let path = file_path_for_callback.clone();
                let heatmap = heatmap_for_callback.clone();

                // Spawn async task to reload data
                glib::spawn_future_local(async move {
                    let max_elements = if tensor.size_bytes > 20 * 1024 * 1024 * 1024 {
                        512 * 256
                    } else if tensor.size_bytes > 5 * 1024 * 1024 * 1024 {
                        1024 * 512
                    } else {
                        2048 * 1024
                    };

                    match tensor.read_tensor_data_with_slice(&path, max_elements, Some(&new_selection)) {
                        Ok(data) => {
                            let (h, w) = new_selection.slice_shape();
                            heatmap.set_data(data, &vec![h, w]);
                            eprintln!("Data reloaded successfully for new slice");
                        }
                        Err(e) => {
                            eprintln!("Error reloading slice data: {}", e);
                        }
                    }
                });
            });

            match tensor_clone.read_tensor_data_with_slice(&file_path_clone, max_elements, slice_selection.as_ref()) {
                Ok(data) => {
                    // Get the shape to display based on slice selection
                    let display_shape = if let Some(ref sel) = slice_selection {
                        let (h, w) = sel.slice_shape();
                        vec![h, w]
                    } else {
                        tensor_clone.dimensions.clone()
                    };

                    heatmap_weak.set_data(data, &display_shape);
                    heatmap_weak.widget().set_visible(true);
                    if let Some(label) = status_weak.upgrade() {
                        label.set_visible(false);
                    }

                    // Enable the toggle button now that data is loaded
                    if let Some(btn) = toggle_btn_weak.upgrade() {
                        btn.set_sensitive(true);

                        // Connect click handler to toggle display mode
                        let heatmap_for_toggle = heatmap_weak.clone();
                        btn.connect_clicked(move |button| {
                            heatmap_for_toggle.toggle_display_mode();
                            // Update button label based on current mode
                            let current_label = button.label().unwrap_or_default();
                            if current_label == "Histogram" {
                                button.set_label("Heatmap");
                            } else {
                                button.set_label("Histogram");
                            }
                        });
                    }
                }
                Err(e) => {
                    if let Some(label) = status_weak.upgrade() {
                        label.set_label(&format!("Error loading tensor: {}", e));
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

        // Draw text labels
        cr.set_source_rgba(1.0, 1.0, 1.0, 0.9);
        cr.set_font_size(10.5);

        // Device name (top left) with available memory
        let name_text = if model_allocated > 0 && available_after > model_allocated {
            format!("{} • {} available", region.name, format_bytes(available_after - model_allocated))
        } else {
            region.name.clone()
        };
        cr.move_to(margin + bar_margin, region.y_pos + 16.0);
        cr.show_text(&name_text).unwrap();

        // Memory stats (top right)
        cr.set_font_size(10.0);
        let stats_text = if model_allocated > 0 {
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

    // Legend items with color boxes - better spacing
    let legend_start_x = margin + 150.0;
    let box_size = 11.0;
    let box_y_offset = 7.5;
    let legend_item_spacing = 105.0;

    // "Currently Used" legend
    cr.set_source_rgba(0.37, 0.62, 0.31, 0.6);
    draw_rounded_rect(cr, legend_start_x, legend_y - box_y_offset, box_size, box_size, 2.0);
    cr.fill().unwrap();

    cr.set_source_rgba(1.0, 1.0, 1.0, 0.75);
    cr.move_to(legend_start_x + box_size + 8.0, legend_y);
    cr.show_text("Currently Used").unwrap();

    // "Model" legend
    let model_x = legend_start_x + legend_item_spacing;
    cr.set_source_rgba(0.46, 0.78, 0.39, 1.0);
    draw_rounded_rect(cr, model_x, legend_y - box_y_offset, box_size, box_size, 2.0);
    cr.fill().unwrap();

    cr.set_source_rgba(1.0, 1.0, 1.0, 0.75);
    cr.move_to(model_x + box_size + 8.0, legend_y);
    cr.show_text("Model").unwrap();

    // "Warning" legend
    let warning_x = model_x + legend_item_spacing - 20.0;
    cr.set_source_rgba(0.95, 0.60, 0.00, 1.0);
    draw_rounded_rect(cr, warning_x, legend_y - box_y_offset, box_size, box_size, 2.0);
    cr.fill().unwrap();

    cr.set_source_rgba(1.0, 1.0, 1.0, 0.75);
    cr.move_to(warning_x + box_size + 8.0, legend_y);
    cr.show_text(">85% Full").unwrap();
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
