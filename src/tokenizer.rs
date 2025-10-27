use gtk::prelude::*;
use gtk::{glib, Box as GtkBox, Orientation, ScrolledWindow, SearchEntry};
use adw::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

use crate::gguf_parser::{GGUFFile, MetadataValue};
use crate::token_display::{TokenDisplay, TokenInfo};

#[derive(Clone)]
pub struct TokenizerPage {
    widget: adw::Bin,
    token_store: Rc<RefCell<Vec<TokenData>>>,
    list_view: gtk::ListView,
    search_entry: SearchEntry,
    test_input: gtk::TextView,
    token_display: TokenDisplay,
    summary_label: gtk::Label,
    debounce_cancelled: Rc<RefCell<bool>>,
}

#[derive(Clone, Debug)]
struct TokenData {
    id: usize,
    token: String,
    score: f32,
    token_type: u32,
}

impl TokenizerPage {
    pub fn new() -> Self {
        // Use a paned layout with token tester on top and vocabulary below
        let paned = gtk::Paned::builder()
            .orientation(Orientation::Vertical)
            .shrink_start_child(false)
            .shrink_end_child(false)
            .resize_start_child(false)
            .resize_end_child(true)
            .build();

        // === Token Tester Section (Top) ===
        let tester_box = GtkBox::new(Orientation::Vertical, 0);
        tester_box.add_css_class("toolbar");

        // Header for tester section
        let tester_header = adw::HeaderBar::builder()
            .show_end_title_buttons(false)
            .show_start_title_buttons(false)
            .build();

        let tester_title = adw::WindowTitle::builder()
            .title("Token Tester")
            .build();
        tester_header.set_title_widget(Some(&tester_title));

        tester_box.append(&tester_header);

        // Tester content in a clamp for better readability
        let tester_content = GtkBox::new(Orientation::Vertical, 18);
        tester_content.set_margin_top(18);
        tester_content.set_margin_bottom(18);
        tester_content.set_margin_start(18);
        tester_content.set_margin_end(18);

        // Input section with proper card styling
        let input_group = adw::PreferencesGroup::builder()
            .title("Input")
            .build();

        let test_input = gtk::TextView::builder()
            .wrap_mode(gtk::WrapMode::WordChar)
            .accepts_tab(false)
            .top_margin(12)
            .bottom_margin(12)
            .left_margin(12)
            .right_margin(12)
            .height_request(80)
            .build();

        // Add placeholder text styling
        let input_buffer = test_input.buffer();
        input_buffer.set_text("Enter text to tokenize...");

        let input_scrolled = ScrolledWindow::builder()
            .hscrollbar_policy(gtk::PolicyType::Never)
            .vscrollbar_policy(gtk::PolicyType::Automatic)
            .child(&test_input)
            .build();
        input_scrolled.add_css_class("card");

        input_group.add(&input_scrolled);
        tester_content.append(&input_group);

        // Output section with custom token visualization
        let output_group = adw::PreferencesGroup::builder()
            .title("Tokenization Result")
            .build();

        // Summary label
        let summary_label = gtk::Label::builder()
            .halign(gtk::Align::Start)
            .css_classes(vec!["dim-label".to_string()])
            .margin_start(12)
            .margin_bottom(6)
            .build();

        let summary_box = GtkBox::new(Orientation::Vertical, 0);
        summary_box.append(&summary_label);

        // Token display widget
        let token_display = TokenDisplay::new();
        token_display.widget().add_css_class("card");

        let output_box = GtkBox::new(Orientation::Vertical, 6);
        output_box.append(&summary_box);
        output_box.append(token_display.widget());

        output_group.add(&output_box);
        tester_content.append(&output_group);

        // Wrap tester content in clamp
        let tester_clamp = adw::Clamp::builder()
            .maximum_size(900)
            .tightening_threshold(600)
            .child(&tester_content)
            .build();

        let tester_scrolled = ScrolledWindow::builder()
            .hscrollbar_policy(gtk::PolicyType::Never)
            .vscrollbar_policy(gtk::PolicyType::Automatic)
            .child(&tester_clamp)
            .build();

        tester_box.append(&tester_scrolled);

        // === Vocabulary List Section (Bottom) ===
        let vocab_box = GtkBox::new(Orientation::Vertical, 0);

        // Header for vocabulary section
        let vocab_header = adw::HeaderBar::builder()
            .show_end_title_buttons(false)
            .show_start_title_buttons(false)
            .build();

        let vocab_title = adw::WindowTitle::builder()
            .title("Vocabulary")
            .build();
        vocab_header.set_title_widget(Some(&vocab_title));

        vocab_box.append(&vocab_header);

        // Search bar for vocabulary
        let search_bar_box = GtkBox::new(Orientation::Horizontal, 0);
        search_bar_box.add_css_class("toolbar");
        search_bar_box.set_margin_top(6);
        search_bar_box.set_margin_bottom(6);
        search_bar_box.set_margin_start(6);
        search_bar_box.set_margin_end(6);

        let search_entry = SearchEntry::builder()
            .placeholder_text("Search tokens...")
            .hexpand(true)
            .build();

        search_bar_box.append(&search_entry);
        vocab_box.append(&search_bar_box);

        // Create token list with better styling
        let token_store = Rc::new(RefCell::new(Vec::new()));

        let model = gio::ListStore::new::<glib::BoxedAnyObject>();
        let selection_model = gtk::NoSelection::new(Some(model.clone()));

        let factory = gtk::SignalListItemFactory::new();
        factory.connect_setup(|_, list_item| {
            let list_item = list_item.downcast_ref::<gtk::ListItem>().unwrap();

            // Create a more detailed row layout
            let row = adw::ActionRow::builder()
                .build();

            // Add a label for the token ID on the left
            let id_label = gtk::Label::builder()
                .width_chars(6)
                .xalign(1.0)
                .build();
            id_label.add_css_class("dim-label");
            id_label.add_css_class("numeric");
            row.add_prefix(&id_label);

            list_item.set_child(Some(&row));
        });

        factory.connect_bind(move |_, list_item| {
            let list_item = list_item.downcast_ref::<gtk::ListItem>().unwrap();
            let item = list_item.item()
                .and_downcast::<glib::BoxedAnyObject>()
                .unwrap();
            let token_data = item.borrow::<TokenData>();

            let row = list_item.child()
                .and_downcast::<adw::ActionRow>()
                .unwrap();

            // Update ID label
            if let Some(id_label) = row.first_child() {
                if let Some(id_label) = id_label.downcast_ref::<gtk::Label>() {
                    id_label.set_text(&format!("{}", token_data.id));
                }
            }

            // Format the token with proper escaping
            let escaped_token = escape_token(&token_data.token);
            row.set_title(&format!("\"{}\"", escaped_token));

            // Show score and type as subtitle with better formatting
            let subtitle = format!("score: {:.4}  •  type: {}",
                token_data.score, format_token_type(token_data.token_type));
            row.set_subtitle(&subtitle);
        });

        let list_view = gtk::ListView::new(Some(selection_model.clone()), Some(factory));

        let vocab_scrolled = ScrolledWindow::builder()
            .hscrollbar_policy(gtk::PolicyType::Never)
            .vscrollbar_policy(gtk::PolicyType::Automatic)
            .child(&list_view)
            .vexpand(true)
            .build();

        vocab_box.append(&vocab_scrolled);

        // Add both sections to paned
        paned.set_start_child(Some(&tester_box));
        paned.set_end_child(Some(&vocab_box));
        paned.set_position(360); // Give more space to tester initially

        // Wrap in bin
        let widget = adw::Bin::new();
        widget.set_child(Some(&paned));

        let page = Self {
            widget,
            token_store: token_store.clone(),
            list_view,
            search_entry: search_entry.clone(),
            test_input: test_input.clone(),
            token_display: token_display.clone(),
            summary_label: summary_label.clone(),
            debounce_cancelled: Rc::new(RefCell::new(false)),
        };

        // Connect search
        let token_store_weak = Rc::downgrade(&token_store);
        let model_clone = model.clone();
        search_entry.connect_search_changed(move |entry| {
            if let Some(store) = token_store_weak.upgrade() {
                let query = entry.text().to_lowercase();
                let tokens = store.borrow();

                model_clone.remove_all();

                for token in tokens.iter() {
                    if query.is_empty() ||
                       token.token.to_lowercase().contains(&query) ||
                       token.id.to_string().contains(&query) {
                        model_clone.append(&glib::BoxedAnyObject::new(token.clone()));
                    }
                }
            }
        });

        // Connect test input with debouncing for better performance
        let token_store_weak = Rc::downgrade(&token_store);
        let token_display_weak = token_display.clone();
        let summary_label_weak = summary_label.clone();
        let debounce_cancelled_weak = page.debounce_cancelled.clone();
        let input_buffer_weak = input_buffer.downgrade();

        input_buffer.connect_changed(move |buffer| {
            // Mark any previous pending timeout as cancelled
            *debounce_cancelled_weak.borrow_mut() = true;

            let store_weak = token_store_weak.clone();
            let display_weak = token_display_weak.clone();
            let label_weak = summary_label_weak.clone();
            let buffer_weak = buffer.downgrade();
            let cancelled_weak = debounce_cancelled_weak.clone();

            // Schedule new update with debouncing (150ms delay)
            glib::timeout_add_local(std::time::Duration::from_millis(150), move || {
                // Check if this timeout was cancelled by a newer input event
                if *cancelled_weak.borrow() {
                    *cancelled_weak.borrow_mut() = false;
                    return glib::ControlFlow::Break;
                }
                if let (Some(store), Some(buffer)) = (store_weak.upgrade(), buffer_weak.upgrade()) {
                    let text = buffer.text(&buffer.start_iter(), &buffer.end_iter(), false);
                    let tokens = store.borrow();

                    // Simple tokenization simulation (greedy longest match)
                    let (token_infos, char_count) = tokenize_to_display(&text, &tokens);

                    // Update summary label
                    if !token_infos.is_empty() {
                        let ratio = char_count as f64 / token_infos.len() as f64;
                        label_weak.set_text(&format!(
                            "{} tokens • {} characters • {:.2} chars/token",
                            token_infos.len(),
                            char_count,
                            ratio
                        ));
                    } else if !text.is_empty() && text != "Enter text to tokenize..." {
                        label_weak.set_text("⚠ No tokens matched (using simplified tokenizer)");
                    } else {
                        label_weak.set_text("");
                    }

                    // Update token display
                    display_weak.set_tokens(token_infos);
                }

                glib::ControlFlow::Break
            });

            // Source ID not needed - we use the cancelled flag instead
        });

        // Clear placeholder on focus
        let input_buffer_for_focus = input_buffer_weak;
        let focus_controller = gtk::EventControllerFocus::new();
        focus_controller.connect_enter(move |_| {
            if let Some(buffer) = input_buffer_for_focus.upgrade() {
                let text = buffer.text(&buffer.start_iter(), &buffer.end_iter(), false);
                if text == "Enter text to tokenize..." {
                    buffer.set_text("");
                }
            }
        });
        test_input.add_controller(focus_controller);

        page
    }

    pub fn widget(&self) -> &adw::Bin {
        &self.widget
    }

    pub fn load_tokenizer(&self, gguf_file: &GGUFFile) {
        let mut tokens = Vec::new();

        // Get tokens array
        if let Some(tokens_array) = gguf_file.metadata.get_array("tokenizer.ggml.tokens") {
            // Get scores (optional)
            let scores = gguf_file.metadata.get_array("tokenizer.ggml.scores");

            // Get token types (optional)
            let token_types = gguf_file.metadata.get_array("tokenizer.ggml.token_type");

            for (i, token_val) in tokens_array.iter().enumerate() {
                if let MetadataValue::String(token_str) = token_val {
                    let score = scores
                        .and_then(|s| s.get(i))
                        .and_then(|v| {
                            if let MetadataValue::Float32(f) = v {
                                Some(*f)
                            } else {
                                None
                            }
                        })
                        .unwrap_or(0.0);

                    let token_type = token_types
                        .and_then(|t| t.get(i))
                        .and_then(|v| {
                            if let MetadataValue::UInt32(t) = v {
                                Some(*t)
                            } else {
                                None
                            }
                        })
                        .unwrap_or(0);

                    tokens.push(TokenData {
                        id: i,
                        token: token_str.clone(),
                        score,
                        token_type,
                    });
                }
            }
        }

        *self.token_store.borrow_mut() = tokens.clone();

        // Update list view
        if let Some(selection_model) = self.list_view.model() {
            if let Some(selection_model) = selection_model.downcast_ref::<gtk::NoSelection>() {
                if let Some(model) = selection_model.model() {
                    let list_store = model.downcast_ref::<gio::ListStore>().unwrap();
                    list_store.remove_all();

                    // Only show first 1000 tokens initially for performance
                    for token in tokens.iter().take(1000) {
                        list_store.append(&glib::BoxedAnyObject::new(token.clone()));
                    }
                }
            }
        }
    }
}

fn escape_token(token: &str) -> String {
    token
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
        .replace('\0', "\\0")
}

fn format_token_type(token_type: u32) -> &'static str {
    match token_type {
        0 => "normal",
        1 => "unknown",
        2 => "control",
        3 => "user defined",
        4 => "unused",
        5 => "byte",
        _ => "other",
    }
}

fn tokenize_to_display(text: &str, tokens: &[TokenData]) -> (Vec<TokenInfo>, usize) {
    if text.is_empty() || text == "Enter text to tokenize..." {
        return (Vec::new(), 0);
    }

    if tokens.is_empty() {
        return (Vec::new(), 0);
    }

    let char_count = text.len();
    let mut remaining = text.to_string();
    let mut token_infos = Vec::new();

    // Very simple greedy tokenization
    while !remaining.is_empty() {
        let mut matched = false;

        // Try to find longest matching token
        for len in (1..=remaining.len()).rev() {
            let substring = &remaining[..len];

            if let Some(token) = tokens.iter().find(|t| t.token == substring) {
                token_infos.push(TokenInfo {
                    index: token_infos.len(),
                    id: token.id,
                    token: token.token.clone(),
                    score: token.score,
                    token_type: token.token_type,
                });
                remaining = remaining[len..].to_string();
                matched = true;
                break;
            }
        }

        if !matched {
            // Skip first character if no match
            remaining = remaining[1..].to_string();
        }
    }

    (token_infos, char_count)
}

fn tokenize_simple(text: &str, tokens: &[TokenData]) -> String {
    if text.is_empty() || text == "Enter text to tokenize..." {
        return String::new();
    }

    if tokens.is_empty() {
        return "No tokenizer data available".to_string();
    }

    let mut result = String::new();
    let mut remaining = text.to_string();
    let mut token_matches = Vec::new();

    // Very simple greedy tokenization
    while !remaining.is_empty() {
        let mut matched = false;

        // Try to find longest matching token
        for len in (1..=remaining.len()).rev() {
            let substring = &remaining[..len];

            if let Some(token) = tokens.iter().find(|t| t.token == substring) {
                token_matches.push((token.id, token.token.clone()));
                remaining = remaining[len..].to_string();
                matched = true;
                break;
            }
        }

        if !matched {
            // Skip first character if no match
            remaining = remaining[1..].to_string();
        }
    }

    if token_matches.is_empty() {
        return "⚠ No tokens matched\n\nNote: This is a simplified tokenizer that uses greedy longest-match.\nActual tokenizers may use BPE, SentencePiece, or other algorithms.".to_string();
    }

    // Summary header
    result.push_str(&format!("📊 Tokenization Summary\n"));
    result.push_str(&format!("─────────────────────\n"));
    result.push_str(&format!("Tokens: {}\n", token_matches.len()));
    result.push_str(&format!("Characters: {}\n", text.len()));
    result.push_str(&format!("Ratio: {:.2} chars/token\n\n", text.len() as f64 / token_matches.len() as f64));

    // Token breakdown
    result.push_str("🔤 Token Breakdown\n");
    result.push_str("──────────────────\n\n");

    for (i, (id, token_str)) in token_matches.iter().enumerate() {
        let escaped = escape_token(token_str);

        // Format with box drawing characters for better visual separation
        if i > 0 {
            result.push_str("│\n");
        }
        result.push_str(&format!("├─ [{:3}] ", i));
        result.push_str(&format!("ID {:5}  ", id));
        result.push_str(&format!("│ \"{}\"", escaped));
        result.push('\n');
    }

    result.push_str("└─────────────────────\n");

    result
}
