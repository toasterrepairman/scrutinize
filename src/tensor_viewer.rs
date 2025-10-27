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
        memory_viz.set_content_height(160);
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

        let model = gio::ListStore::new::<glib::BoxedAnyObject>();
        let selection_model = gtk::NoSelection::new(Some(model.clone()));

        let factory = gtk::SignalListItemFactory::new();
        factory.connect_setup(|_, list_item| {
            let row = adw::ActionRow::builder().build();

            let type_label = gtk::Label::builder()
                .margin_end(8)
                .css_classes(["dim-label"])
                .build();
            row.add_suffix(&type_label);

            let list_item = list_item.downcast_ref::<gtk::ListItem>().unwrap();
            list_item.set_child(Some(&row));
        });

        factory.connect_bind(move |_, list_item| {
            let list_item = list_item.downcast_ref::<gtk::ListItem>().unwrap();
            let item = list_item.item()
                .and_downcast::<glib::BoxedAnyObject>()
                .unwrap();
            let tensor = item.borrow::<TensorInfo>();

            let row = list_item.child()
                .and_downcast::<adw::ActionRow>()
                .unwrap();

            row.set_title(&tensor.name);

            let subtitle = format!("Shape: {} • Size: {}",
                tensor.shape_string(),
                format_bytes(tensor.size_bytes));
            row.set_subtitle(&subtitle);

            // Update type label
            if let Some(type_label) = row.first_child() {
                let mut current = Some(type_label);
                while let Some(widget) = current {
                    if let Ok(label) = widget.clone().downcast::<gtk::Label>() {
                        label.set_text(tensor.dtype.name());
                        break;
                    }
                    current = widget.next_sibling();
                }
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

        // Trigger redraw of memory visualization
        self.memory_viz.queue_draw();
    }
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
