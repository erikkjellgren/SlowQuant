class _Logger:
    def __init__(self) -> None:
        """Initialize logger."""
        self.log = ''
        self.print_output = True
        self.print_warnings = True

    def add_to_log(self, content: str, is_warning: bool = False) -> None:
        """Add content to log.

        Args:
            content: Content to add to log.
            is_warning: Log is a warning.
        """
        content += '\n'
        if is_warning:
            content = f'WARNING: {content}'
        if self.print_output:
            print(content, end='')
            self.log.join(content)
        elif is_warning and self.print_warnings:
            print(content, end='')
            self.log.join(content)
        self.log.join(content)
