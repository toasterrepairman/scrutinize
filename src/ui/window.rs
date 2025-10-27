use gtk::prelude::*;
use gtk::{ScrolledWindow, Box as GtkBox, Orientation};
use adw::prelude::*;
use std::path::Path;
use std::cell::RefCell;
use std::rc::Rc;

use crate::gguf_parser::GGUFFile;
use crate::tokenizer::TokenizerPage;
use crate::tensor_viewer::TensorPage;

#[derive(Clone)]
pub struct GGUFWindow {
    window: adw::ApplicationWindow,
    header_bar: adw::HeaderBar,
    toolbar_view: adw::ToolbarView,
    view_stack: adw::ViewStack,
    view_switcher_bar: adw::ViewSwitcherBar,
    overview_page: ScrolledWindow,
    tokenizer_page: TokenizerPage,
    tensor_page: TensorPage,
    gguf_data: Rc<RefCell<Option<GGUFFile>>>,
}

impl GGUFWindow {
    pub fn new(app: &adw::Application) -> Self {
        let window = adw::ApplicationWindow::builder()
            .application(app)
            .title("Scrutinize")
            .default_width(500)
            .default_height(800)
            .build();

        // Create header bar
        let header_bar = adw::HeaderBar::builder().build();

        let open_button = gtk::Button::builder()
            .icon_name("document-open-symbolic")
            .tooltip_text("Open GGUF file")
            .build();

        header_bar.pack_start(&open_button);

        // Create view stack for different pages
        let view_stack = adw::ViewStack::builder()
            .vexpand(true)
            .hexpand(true)
            .build();

        // Create overview page (initially empty)
        let overview_page = ScrolledWindow::builder()
            .hscrollbar_policy(gtk::PolicyType::Never)
            .vscrollbar_policy(gtk::PolicyType::Automatic)
            .build();

        let empty_status = adw::StatusPage::builder()
            .icon_name("document-open-symbolic")
            .title("No Model Loaded")
            .description("Open a GGUF file to view its metadata and structure")
            .build();

        overview_page.set_child(Some(&empty_status));

        // Create tokenizer page
        let tokenizer_page = TokenizerPage::new();

        // Create tensor page
        let tensor_page = TensorPage::new();

        // Add pages to view stack
        view_stack.add_titled(&overview_page, Some("overview"), "Overview");
        view_stack.add_titled(tokenizer_page.widget(), Some("tokenizer"), "Tokenizer");
        view_stack.add_titled(tensor_page.widget(), Some("tensors"), "Tensors");

        // Create view switcher bar
        let view_switcher_bar = adw::ViewSwitcherBar::builder()
            .stack(&view_stack)
            .reveal(true)
            .build();

        // Create toolbar view
        let toolbar_view = adw::ToolbarView::builder().build();
        toolbar_view.add_top_bar(&header_bar);
        toolbar_view.add_bottom_bar(&view_switcher_bar);
        toolbar_view.set_content(Some(&view_stack));

        window.set_content(Some(&toolbar_view));

        let gguf_data = Rc::new(RefCell::new(None));

        let window_clone = Self {
            window: window.clone(),
            header_bar,
            toolbar_view,
            view_stack,
            view_switcher_bar,
            overview_page: overview_page.clone(),
            tokenizer_page: tokenizer_page.clone(),
            tensor_page: tensor_page.clone(),
            gguf_data: gguf_data.clone(),
        };

        // Set up keyboard shortcuts
        let quit_action = gio::SimpleAction::new("quit", None);
        let window_weak_quit = window.downgrade();
        quit_action.connect_activate(move |_, _| {
            if let Some(window) = window_weak_quit.upgrade() {
                window.close();
            }
        });
        window.add_action(&quit_action);

        // Add Ctrl+Q accelerator
        if let Some(app) = window.application() {
            app.set_accels_for_action("win.quit", &["<Ctrl>Q"]);
        }

        // Connect open button
        let window_weak = window_clone.clone();
        open_button.connect_clicked(move |_| {
            window_weak.open_file_dialog();
        });

        window_clone
    }

    pub fn present(&self) {
        self.window.present();
    }

    fn open_file_dialog(&self) {
        let dialog = gtk::FileDialog::builder()
            .title("Open GGUF File")
            .modal(true)
            .build();

        // Add file filter for GGUF files
        let filter = gtk::FileFilter::new();
        filter.add_pattern("*.gguf");
        filter.set_name(Some("GGUF Files"));

        let filters = gio::ListStore::new::<gtk::FileFilter>();
        filters.append(&filter);
        dialog.set_filters(Some(&filters));

        let window_weak = self.clone();
        dialog.open(Some(&self.window), None::<&gio::Cancellable>, move |result| {
            if let Ok(file) = result {
                if let Some(path) = file.path() {
                    window_weak.load_file(&path);
                }
            }
        });
    }

    pub fn load_file(&self, path: &Path) {
        // Show loading state
        let loading_status = adw::StatusPage::builder()
            .icon_name("document-open-symbolic")
            .title("Loading Model...")
            .description(path.display().to_string())
            .build();

        self.overview_page.set_child(Some(&loading_status));

        // Load GGUF file
        match GGUFFile::load(path) {
            Ok(gguf_file) => {
                *self.gguf_data.borrow_mut() = Some(gguf_file.clone());

                // Update window title
                let filename = path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("GGUF File");
                self.window.set_title(Some(&format!("{} - Scrutinize", filename)));

                // Build overview page
                self.build_overview_page(&gguf_file);

                // Update tokenizer page
                self.tokenizer_page.load_tokenizer(&gguf_file);

                // Update tensor page
                self.tensor_page.set_file_path(path.to_path_buf());
                self.tensor_page.load_tensors(&gguf_file);
            }
            Err(e) => {
                let error_status = adw::StatusPage::builder()
                    .icon_name("dialog-error-symbolic")
                    .title("Failed to Load Model")
                    .description(format!("Error: {}", e))
                    .build();

                self.overview_page.set_child(Some(&error_status));
            }
        }
    }

    fn build_overview_page(&self, gguf_file: &GGUFFile) {
        let content = GtkBox::new(Orientation::Vertical, 0);
        content.set_margin_top(12);
        content.set_margin_bottom(12);
        content.set_margin_start(12);
        content.set_margin_end(12);

        // Model header
        let header_group = adw::PreferencesGroup::builder()
            .title("Model Information")
            .build();

        if let Some(name) = gguf_file.metadata.get_string("general.name") {
            let name_row = adw::ActionRow::builder()
                .title("Model Name")
                .subtitle(&name)
                .build();
            header_group.add(&name_row);
        }

        if let Some(arch) = gguf_file.metadata.get_string("general.architecture") {
            let arch_row = adw::ActionRow::builder()
                .title("Architecture")
                .subtitle(&arch)
                .build();
            header_group.add(&arch_row);
        }

        if let Some(basename) = gguf_file.metadata.get_string("general.basename") {
            let basename_row = adw::ActionRow::builder()
                .title("Base Model")
                .subtitle(&basename)
                .build();
            header_group.add(&basename_row);
        }

        if let Some(size_label) = gguf_file.metadata.get_string("general.size_label") {
            let size_row = adw::ActionRow::builder()
                .title("Size")
                .subtitle(&size_label)
                .build();
            header_group.add(&size_row);
        }

        if let Some(file_type) = gguf_file.metadata.get_u32("general.file_type") {
            let file_type_str = format_file_type(file_type);
            let type_row = adw::ActionRow::builder()
                .title("Quantization")
                .subtitle(&file_type_str)
                .build();
            header_group.add(&type_row);
        }

        content.append(&header_group);

        // MoE (Mixture of Experts) widget - only show for MoE models
        if let Some(arch) = gguf_file.metadata.get_string("general.architecture") {
            self.build_moe_widget_if_applicable(&content, gguf_file, &arch);
        }

        // Architecture parameters
        if let Some(arch) = gguf_file.metadata.get_string("general.architecture") {
            let arch_group = adw::PreferencesGroup::builder()
                .title("Architecture Parameters")
                .margin_top(24)
                .build();

            // Context length
            if let Some(ctx_len) = gguf_file.metadata.get_u32(&format!("{}.context_length", arch)) {
                let ctx_row = adw::ActionRow::builder()
                    .title("Context Length")
                    .subtitle(&format!("{} tokens", ctx_len))
                    .build();
                arch_group.add(&ctx_row);
            }

            // Embedding length
            if let Some(emb_len) = gguf_file.metadata.get_u32(&format!("{}.embedding_length", arch)) {
                let emb_row = adw::ActionRow::builder()
                    .title("Embedding Dimension")
                    .subtitle(&format!("{}", emb_len))
                    .build();
                arch_group.add(&emb_row);
            }

            // Block count (layers)
            if let Some(blocks) = gguf_file.metadata.get_u32(&format!("{}.block_count", arch)) {
                let blocks_row = adw::ActionRow::builder()
                    .title("Number of Layers")
                    .subtitle(&format!("{}", blocks))
                    .build();
                arch_group.add(&blocks_row);
            }

            // Attention heads
            if let Some(heads) = gguf_file.metadata.get_u32(&format!("{}.attention.head_count", arch)) {
                let heads_row = adw::ActionRow::builder()
                    .title("Attention Heads")
                    .subtitle(&format!("{}", heads))
                    .build();
                arch_group.add(&heads_row);
            }

            // KV heads (for GQA)
            if let Some(kv_heads) = gguf_file.metadata.get_u32(&format!("{}.attention.head_count_kv", arch)) {
                let kv_row = adw::ActionRow::builder()
                    .title("KV Cache Heads")
                    .subtitle(&format!("{}", kv_heads))
                    .build();
                arch_group.add(&kv_row);
            }

            // Feed forward length
            if let Some(ff_len) = gguf_file.metadata.get_u32(&format!("{}.feed_forward_length", arch)) {
                let ff_row = adw::ActionRow::builder()
                    .title("FFN Intermediate Size")
                    .subtitle(&format!("{}", ff_len))
                    .build();
                arch_group.add(&ff_row);
            }

            // RoPE dimension
            if let Some(rope_dim) = gguf_file.metadata.get_u32(&format!("{}.rope.dimension_count", arch)) {
                let rope_row = adw::ActionRow::builder()
                    .title("RoPE Dimensions")
                    .subtitle(&format!("{}", rope_dim))
                    .build();
                arch_group.add(&rope_row);
            }

            // RoPE frequency base
            if let Some(rope_freq) = gguf_file.metadata.get_f32(&format!("{}.rope.freq_base", arch)) {
                let freq_row = adw::ActionRow::builder()
                    .title("RoPE Frequency Base")
                    .subtitle(&format!("{:.1}", rope_freq))
                    .build();
                arch_group.add(&freq_row);
            }

            content.append(&arch_group);
        }

        // File statistics
        let stats_group = adw::PreferencesGroup::builder()
            .title("File Statistics")
            .margin_top(24)
            .build();

        // Total parameters (computed from tensors)
        let total_params = gguf_file.compute_total_parameters();
        let params_row = adw::ActionRow::builder()
            .title("Total Parameters")
            .subtitle(&format_parameter_count(total_params))
            .build();
        stats_group.add(&params_row);

        // File size
        let file_size = gguf_file.file_size;
        let size_row = adw::ActionRow::builder()
            .title("File Size")
            .subtitle(&format_bytes(file_size))
            .build();
        stats_group.add(&size_row);

        // Number of tensors
        let tensor_count = gguf_file.tensors.len();
        let tensor_row = adw::ActionRow::builder()
            .title("Tensor Count")
            .subtitle(&format!("{}", tensor_count))
            .build();
        stats_group.add(&tensor_row);

        // Vocabulary size
        if let Some(vocab_size) = gguf_file.get_vocab_size() {
            let vocab_row = adw::ActionRow::builder()
                .title("Vocabulary Size")
                .subtitle(&format!("{} tokens", vocab_size))
                .build();
            stats_group.add(&vocab_row);
        }

        content.append(&stats_group);

        // Wrap in a clamp for better appearance on wide screens
        let clamp = adw::Clamp::builder()
            .maximum_size(800)
            .tightening_threshold(600)
            .child(&content)
            .build();

        self.overview_page.set_child(Some(&clamp));
    }

    fn build_moe_widget_if_applicable(&self, content: &GtkBox, gguf_file: &GGUFFile, arch: &str) {
        // Check if this is an MoE model by looking for expert_count metadata
        let expert_count = gguf_file.metadata.get_u32(&format!("{}.expert_count", arch));
        let expert_used_count = gguf_file.metadata.get_u32(&format!("{}.expert_used_count", arch));

        if let Some(total_experts) = expert_count {
            // Validate that we have a reasonable expert count
            if total_experts == 0 {
                return;
            }
            let moe_group = adw::PreferencesGroup::builder()
                .title("Mixture of Experts (MoE)")
                .margin_top(24)
                .build();

            // Create a custom widget container for the MoE visualization
            let moe_box = GtkBox::new(Orientation::Vertical, 12);
            moe_box.set_margin_top(12);
            moe_box.set_margin_bottom(12);
            moe_box.set_margin_start(12);
            moe_box.set_margin_end(12);

            // Expert counts row
            let active_experts = expert_used_count.unwrap_or(2); // Default to 2 if not specified

            let expert_info_box = GtkBox::new(Orientation::Horizontal, 12);
            expert_info_box.set_halign(gtk::Align::Center);

            // Total experts label
            let total_label_box = GtkBox::new(Orientation::Vertical, 4);
            let total_value = gtk::Label::new(Some(&format!("{}", total_experts)));
            total_value.add_css_class("title-1");
            let total_caption = gtk::Label::new(Some("Total Experts"));
            total_caption.add_css_class("dim-label");
            total_caption.add_css_class("caption");
            total_label_box.append(&total_value);
            total_label_box.append(&total_caption);

            // Separator
            let separator = gtk::Separator::new(Orientation::Vertical);

            // Active experts label
            let active_label_box = GtkBox::new(Orientation::Vertical, 4);
            let active_value = gtk::Label::new(Some(&format!("{}", active_experts)));
            active_value.add_css_class("title-1");
            active_value.add_css_class("accent");
            let active_caption = gtk::Label::new(Some("Active per Token"));
            active_caption.add_css_class("dim-label");
            active_caption.add_css_class("caption");
            active_label_box.append(&active_value);
            active_label_box.append(&active_caption);

            expert_info_box.append(&total_label_box);
            expert_info_box.append(&separator);
            expert_info_box.append(&active_label_box);

            moe_box.append(&expert_info_box);

            // Visual representation (simplified bar showing ratio)
            let visual_box = GtkBox::new(Orientation::Vertical, 6);

            let progress_bar_container = GtkBox::new(Orientation::Horizontal, 0);
            progress_bar_container.set_hexpand(true);

            let progress = gtk::ProgressBar::new();
            // Clamp the fraction to valid range [0.0, 1.0]
            let fraction = (active_experts as f64 / total_experts as f64).clamp(0.0, 1.0);
            progress.set_fraction(fraction);
            progress.set_show_text(false);
            progress.set_hexpand(true);
            progress_bar_container.append(&progress);

            let ratio_label = gtk::Label::new(Some(&format!("{}/{} experts active", active_experts, total_experts)));
            ratio_label.add_css_class("caption");
            ratio_label.add_css_class("dim-label");
            ratio_label.set_halign(gtk::Align::Center);

            visual_box.append(&progress_bar_container);
            visual_box.append(&ratio_label);

            moe_box.append(&visual_box);

            // Model size information
            if let Some(size_label) = gguf_file.metadata.get_string("general.size_label") {
                let size_info_box = GtkBox::new(Orientation::Vertical, 4);
                size_info_box.set_margin_top(6);

                // Calculate estimated active size
                let file_size = gguf_file.file_size;
                let estimated_active_size = calculate_active_expert_size(
                    file_size,
                    total_experts,
                    active_experts,
                );

                let size_row = GtkBox::new(Orientation::Horizontal, 8);
                size_row.set_halign(gtk::Align::Center);

                let full_size_label = gtk::Label::new(Some(&format!("Full Model: {}", size_label)));
                full_size_label.add_css_class("caption");

                let dot_separator = gtk::Label::new(Some("•"));
                dot_separator.add_css_class("caption");
                dot_separator.add_css_class("dim-label");

                let active_size_label = gtk::Label::new(Some(&format!(
                    "Active: ~{}",
                    format_bytes(estimated_active_size)
                )));
                active_size_label.add_css_class("caption");
                active_size_label.add_css_class("accent");

                size_row.append(&full_size_label);
                size_row.append(&dot_separator);
                size_row.append(&active_size_label);

                size_info_box.append(&size_row);
                moe_box.append(&size_info_box);
            }

            // Wrap in a frame for visual grouping
            let frame = gtk::Frame::new(None);
            frame.set_child(Some(&moe_box));

            // Use ActionRow to embed the custom widget
            let moe_container = adw::PreferencesGroup::builder()
                .title("Mixture of Experts Configuration")
                .margin_top(24)
                .build();

            moe_container.add(&frame);
            content.append(&moe_container);
        }
    }
}

fn calculate_active_expert_size(total_size: u64, total_experts: u32, active_experts: u32) -> u64 {
    // This is a rough estimation. In reality, MoE models have:
    // - Shared parameters (embeddings, layer norms, output layer)
    // - Expert-specific parameters (FFN layers)
    // We estimate that ~70% of the model size is in expert-specific parameters
    // and ~30% is shared across all experts

    // Guard against division by zero
    if total_experts == 0 {
        return total_size;
    }

    let shared_portion = (total_size as f64 * 0.30) as u64;
    let expert_portion = (total_size as f64 * 0.70) as u64;
    let per_expert_size = expert_portion / total_experts as u64;
    let active_expert_size = per_expert_size * active_experts as u64;

    shared_portion + active_expert_size
}

fn format_file_type(file_type: u32) -> String {
    match file_type {
        0 => "F32 (All 32-bit floats)".to_string(),
        1 => "F16 (All 16-bit floats)".to_string(),
        2 => "Q4_0 (4-bit quantization)".to_string(),
        3 => "Q4_1 (4-bit quantization)".to_string(),
        7 => "Q8_0 (8-bit quantization)".to_string(),
        _ => format!("Type {} (Mixed quantization)", file_type),
    }
}

fn format_parameter_count(count: u64) -> String {
    if count >= 1_000_000_000 {
        format!("{:.2}B", count as f64 / 1_000_000_000.0)
    } else if count >= 1_000_000 {
        format!("{:.2}M", count as f64 / 1_000_000.0)
    } else if count >= 1_000 {
        format!("{:.2}K", count as f64 / 1_000.0)
    } else {
        format!("{}", count)
    }
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
        format!("{} bytes", bytes)
    }
}
