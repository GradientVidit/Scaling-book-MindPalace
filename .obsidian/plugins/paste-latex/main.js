const { Plugin, MarkdownView } = require("obsidian");

module.exports = class PasteLatexPlugin extends Plugin {

    async pasteWrapped(prefix, suffix) {
        const clipboard = await navigator.clipboard.readText();

        const editor = this.app.workspace
            .getActiveViewOfType(MarkdownView)
            ?.editor;

        if (!editor) return;

        editor.replaceSelection(`${prefix}${clipboard}${suffix}`);
    }

    async onload() {

        this.addCommand({
            id: "paste-display-latex",
            name: "Paste clipboard as display math",
            callback: () => this.pasteWrapped("$$\n", "\n$$")
        });

        this.addCommand({
            id: "paste-inline-latex",
            name: "Paste clipboard as inline math",
            callback: () => this.pasteWrapped("$", "$")
        });

    }

};
