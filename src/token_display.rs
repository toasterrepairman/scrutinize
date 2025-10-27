use gtk::prelude::*;
use gtk::{Box as GtkBox, Orientation, ScrolledWindow};

#[derive(Clone, Debug)]
pub struct TokenInfo {
    pub index: usize,
    pub id: usize,
    pub token: String,
    pub score: f32,
    pub token_type: u32,
}

#[derive(Clone)]
pub struct TokenDisplay {
    widget: ScrolledWindow,
    flow_box: gtk::FlowBox,
}

impl TokenDisplay {
    pub fn new() -> Self {
        let flow_box = gtk::FlowBox::builder()
            .selection_mode(gtk::SelectionMode::None)
            .homogeneous(false)
            .column_spacing(6)
            .row_spacing(6)
            .margin_top(12)
            .margin_bottom(12)
            .margin_start(12)
            .margin_end(12)
            .max_children_per_line(30)
            .build();

        let scrolled = ScrolledWindow::builder()
            .hscrollbar_policy(gtk::PolicyType::Never)
            .vscrollbar_policy(gtk::PolicyType::Automatic)
            .child(&flow_box)
            .vexpand(true)
            .min_content_height(120)
            .build();

        Self {
            widget: scrolled,
            flow_box,
        }
    }

    pub fn widget(&self) -> &ScrolledWindow {
        &self.widget
    }

    pub fn clear(&self) {
        while let Some(child) = self.flow_box.first_child() {
            self.flow_box.remove(&child);
        }
    }

    pub fn set_tokens(&self, tokens: Vec<TokenInfo>) {
        self.clear();

        if tokens.is_empty() {
            // Show empty state
            let empty_label = gtk::Label::builder()
                .label("⚠ No tokens matched")
                .css_classes(vec!["dim-label".to_string()])
                .build();
            self.flow_box.append(&empty_label);
            return;
        }

        for token_info in tokens {
            let token_widget = self.create_token_widget(&token_info);
            self.flow_box.append(&token_widget);
        }
    }

    fn create_token_widget(&self, token_info: &TokenInfo) -> gtk::Widget {
        // Main container for the token
        let token_box = GtkBox::new(Orientation::Vertical, 2);
        token_box.add_css_class("card");
        token_box.set_margin_top(2);
        token_box.set_margin_bottom(2);
        token_box.set_margin_start(2);
        token_box.set_margin_end(2);

        // Token content with colored background based on type
        let content_box = GtkBox::new(Orientation::Horizontal, 6);
        content_box.set_margin_top(6);
        content_box.set_margin_bottom(6);
        content_box.set_margin_start(8);
        content_box.set_margin_end(8);

        // Token text with escaping
        let escaped_token = escape_token(&token_info.token);
        let token_label = gtk::Label::builder()
            .label(&format!("\"{}\"", escaped_token))
            .use_markup(false)
            .selectable(true)
            .css_classes(vec!["monospace".to_string()])
            .build();

        // Apply color based on token type
        match token_info.token_type {
            0 => token_box.add_css_class("token-normal"),      // Normal tokens
            1 => token_box.add_css_class("token-unknown"),     // Unknown
            2 => token_box.add_css_class("token-control"),     // Control
            3 => token_box.add_css_class("token-user"),        // User defined
            4 => token_box.add_css_class("token-unused"),      // Unused
            5 => token_box.add_css_class("token-byte"),        // Byte
            _ => token_box.add_css_class("token-other"),       // Other
        }

        content_box.append(&token_label);
        token_box.append(&content_box);

        // Metadata row (token ID and index)
        let meta_box = GtkBox::new(Orientation::Horizontal, 8);
        meta_box.set_margin_start(8);
        meta_box.set_margin_end(8);
        meta_box.set_margin_bottom(6);
        meta_box.set_halign(gtk::Align::Center);

        let index_label = gtk::Label::builder()
            .label(&format!("[{}]", token_info.index))
            .css_classes(vec!["dim-label".to_string(), "caption".to_string()])
            .build();

        let id_label = gtk::Label::builder()
            .label(&format!("ID:{}", token_info.id))
            .css_classes(vec!["dim-label".to_string(), "caption".to_string(), "numeric".to_string()])
            .build();

        meta_box.append(&index_label);
        meta_box.append(&id_label);
        token_box.append(&meta_box);

        // Add tooltip with full details
        let tooltip_text = format!(
            "Token: \"{}\"\nID: {}\nIndex: {}\nScore: {:.4}\nType: {}",
            escaped_token,
            token_info.id,
            token_info.index,
            token_info.score,
            format_token_type(token_info.token_type)
        );
        token_box.set_tooltip_text(Some(&tooltip_text));

        token_box.upcast()
    }

    pub fn set_summary(&self, _token_count: usize, _char_count: usize) {
        // This could be displayed in a header bar or status area
        // For now, we'll just use it for tooltip or future enhancements
    }
}

fn escape_token(token: &str) -> String {
    let mut result = String::new();
    for ch in token.chars() {
        match ch {
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            '\0' => result.push_str("\\0"),
            ' ' => result.push_str("␣"),  // Visible space character
            c if c.is_control() => result.push_str(&format!("\\u{{{:04x}}}", c as u32)),
            c => result.push(c),
        }
    }
    result
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
