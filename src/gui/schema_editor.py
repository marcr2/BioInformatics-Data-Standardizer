"""
Schema Editor for BIDS GUI

Custom CSV schema builder with save/load functionality.
"""

import dearpygui.dearpygui as dpg
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any
import json

# Lazy imports
SchemaManager = None
Schema = None
ColumnDefinition = None
ColumnType = None
TransformType = None


def get_schema_classes():
    """Lazy import of schema classes."""
    global SchemaManager, Schema, ColumnDefinition, ColumnType, TransformType
    if SchemaManager is None:
        from ..schema_manager import (
            SchemaManager as SM,
            Schema as S,
            ColumnDefinition as CD,
            ColumnType as CT,
            TransformType as TT
        )
        SchemaManager = SM
        Schema = S
        ColumnDefinition = CD
        ColumnType = CT
        TransformType = TT
    return SchemaManager, Schema, ColumnDefinition, ColumnType, TransformType


class SchemaEditor:
    """
    Custom schema editor panel.
    
    Features:
    - Create new schemas
    - Define output columns
    - Set data types and transformations
    - Map source columns
    - Save/load schemas
    """
    
    COLUMN_TYPES = ["string", "int", "float", "bool", "date", "datetime"]
    TRANSFORMS = ["none", "uppercase", "lowercase", "trim", "log2", "log10", "abs", "round", "negate",
                  "fdr_benjamini_hochberg", "fdr_benjamini_yekutieli", "bonferroni", "holm_bonferroni"]
    
    # Statistical test transforms
    STAT_TRANSFORMS = ["fdr_benjamini_hochberg", "fdr_benjamini_yekutieli", "bonferroni", "holm_bonferroni"]
    
    def __init__(self, on_schema_changed: Optional[Callable] = None, get_available_columns: Optional[Callable[[], List[str]]] = None):
        """
        Initialize SchemaEditor.
        
        Args:
            on_schema_changed: Callback when schema changes
            get_available_columns: Callback to get list of available column names from current dataset
        """
        self.on_schema_changed = on_schema_changed
        self.get_available_columns = get_available_columns or (lambda: [])
        self.schema_manager = None
        self.current_schema = None
        self.column_rows: List[Dict[str, Any]] = []
        
        # UI tags
        self.schema_name_tag: Optional[int] = None
        self.schema_desc_tag: Optional[int] = None
        self.columns_container_tag: Optional[int] = None
        self.schema_combo_tag: Optional[int] = None
    
    def create(self) -> None:
        """Create the schema editor UI."""
        SM, S, CD, CT, TT = get_schema_classes()
        self.schema_manager = SM("schemas")
        
        # Schema selection
        with dpg.group(horizontal=True):
            dpg.add_text("Schema:", color=(100, 149, 237))
            
            schemas = self.schema_manager.list_schemas()
            self.schema_combo_tag = dpg.add_combo(
                items=schemas,
                default_value=schemas[0] if schemas else "",
                callback=self._on_schema_combo_changed,
                width=200
            )
            
            dpg.add_spacer(width=10)
            dpg.add_button(label="New", callback=self.new_schema, width=60)
            dpg.add_button(label="Save", callback=self.save_schema, width=60)
            dpg.add_button(label="Delete", callback=self._delete_schema, width=60)
        
        dpg.add_spacer(height=10)
        dpg.add_separator()
        dpg.add_spacer(height=10)
        
        # Schema info
        dpg.add_text("Schema Name:", color=(149, 165, 166))
        self.schema_name_tag = dpg.add_input_text(
            default_value="",
            width=-1,
            callback=self._on_name_changed
        )
        
        dpg.add_spacer(height=5)
        
        dpg.add_text("Description:", color=(149, 165, 166))
        self.schema_desc_tag = dpg.add_input_text(
            default_value="",
            width=-1,
            multiline=True,
            height=60
        )
        
        dpg.add_spacer(height=10)
        dpg.add_separator()
        dpg.add_spacer(height=10)
        
        # Columns section
        with dpg.group(horizontal=True):
            dpg.add_text("Output Columns", color=(100, 149, 237))
            dpg.add_spacer(width=-1)
            dpg.add_button(
                label="+ Add Column",
                callback=self._add_column_row,
                width=100
            )
        
        dpg.add_spacer(height=10)
        
        # Column headers
        with dpg.group(horizontal=True):
            dpg.add_text("Name", color=(149, 165, 166))
            dpg.add_spacer(width=105)
            dpg.add_text("Type", color=(149, 165, 166))
            dpg.add_spacer(width=55)
            dpg.add_text("Source", color=(149, 165, 166))
            dpg.add_spacer(width=75)
            dpg.add_text("Transform", color=(149, 165, 166))
            dpg.add_spacer(width=50)
            dpg.add_text("Req", color=(149, 165, 166))
        
        # Scrollable columns container
        with dpg.child_window(height=300, border=True) as container:
            self.columns_container_tag = container
        
        dpg.add_spacer(height=10)
        
        # Apply button
        with dpg.group(horizontal=True):
            dpg.add_spacer(width=-150)
            dpg.add_button(
                label="Apply Schema",
                callback=self._apply_schema,
                width=140
            )
        
        # Note: Don't auto-load schema here - the app will call load_schema
        # after all UI components are created to avoid callback issues
    
    def _add_column_row(self, sender=None, app_data=None, default_values: Dict = None) -> None:
        """Add a new column definition row."""
        row_id = len(self.column_rows)
        
        defaults = default_values or {
            "name": f"Column{row_id + 1}",
            "type": "string",
            "source": "auto",
            "transform": "none",
            "required": True,
            "stat_test_source_column": None,
            "stat_test_alpha": 0.05
        }
        
        with dpg.group(horizontal=True, parent=self.columns_container_tag) as row_group:
            # Column name
            name_input = dpg.add_input_text(
                default_value=defaults["name"],
                width=120
            )
            
            # Type combo
            type_combo = dpg.add_combo(
                items=self.COLUMN_TYPES,
                default_value=defaults["type"],
                width=80
            )
            
            # Source column
            source_input = dpg.add_input_text(
                default_value=defaults["source"],
                width=100
            )
            
            # Transform
            transform_combo = dpg.add_combo(
                items=self.TRANSFORMS,
                default_value=defaults["transform"],
                width=120,
                callback=lambda s, a, u: self._on_transform_changed(s, a, u),
                user_data=row_id
            )
            
            # Required checkbox
            required_check = dpg.add_checkbox(
                default_value=defaults["required"]
            )
            
            # Delete button
            delete_btn = dpg.add_button(
                label="X",
                width=25,
                callback=lambda s, a, u: self._remove_column_row(u),
                user_data=row_id
            )
        
        # Statistical test configuration group (initially hidden)
        stat_test_group = None
        stat_source_combo = None
        stat_alpha_input = None
        
        # Create statistical test UI group (will be shown/hidden based on transform)
        with dpg.group(parent=self.columns_container_tag) as stat_test_group:
            # Add spacing to align with row above
            dpg.add_spacer(width=305, height=0)
            
            # Source column selector for p-values
            available_cols = self.get_available_columns()
            stat_source_combo = dpg.add_combo(
                items=available_cols if available_cols else ["No columns available"],
                default_value=defaults.get("stat_test_source_column", ""),
                width=150,
                label="P-value column:",
                enabled=len(available_cols) > 0
            )
            
            dpg.add_spacer(width=10)
            
            # Alpha input
            stat_alpha_input = dpg.add_input_float(
                default_value=defaults.get("stat_test_alpha", 0.05),
                width=80,
                label="Alpha:",
                min_value=0.0,
                max_value=1.0,
                format="%.3f"
            )
        
        # Initially hide if not a statistical test
        if defaults["transform"] not in self.STAT_TRANSFORMS:
            dpg.hide_item(stat_test_group)
        
        self.column_rows.append({
            "id": row_id,
            "group": row_group,
            "name": name_input,
            "type": type_combo,
            "source": source_input,
            "transform": transform_combo,
            "required": required_check,
            "stat_test_group": stat_test_group,
            "stat_source_combo": stat_source_combo,
            "stat_alpha_input": stat_alpha_input
        })
    
    def _on_transform_changed(self, sender, app_data, user_data) -> None:
        """Handle transform combo change - show/hide statistical test UI."""
        row_id = user_data
        transform_value = app_data
        
        # Find the row
        for row in self.column_rows:
            if row["id"] == row_id:
                if transform_value in self.STAT_TRANSFORMS:
                    # Show statistical test UI
                    dpg.show_item(row["stat_test_group"])
                    # Update available columns
                    available_cols = self.get_available_columns()
                    if available_cols:
                        dpg.configure_item(row["stat_source_combo"], items=available_cols, enabled=True)
                    else:
                        dpg.configure_item(row["stat_source_combo"], items=["No columns available"], enabled=False)
                else:
                    # Hide statistical test UI
                    dpg.hide_item(row["stat_test_group"])
                break
    
    def _remove_column_row(self, row_id: int) -> None:
        """Remove a column row."""
        for i, row in enumerate(self.column_rows):
            if row["id"] == row_id:
                dpg.delete_item(row["group"])
                # Also delete statistical test group if it exists
                if "stat_test_group" in row and row["stat_test_group"]:
                    try:
                        dpg.delete_item(row["stat_test_group"])
                    except Exception:
                        pass
                self.column_rows.pop(i)
                break
    
    def _clear_column_rows(self) -> None:
        """Clear all column rows."""
        for row in self.column_rows:
            try:
                dpg.delete_item(row["group"])
                # Also delete statistical test group if it exists
                if "stat_test_group" in row and row["stat_test_group"]:
                    try:
                        dpg.delete_item(row["stat_test_group"])
                    except Exception:
                        pass
            except Exception:
                pass
        self.column_rows = []
    
    def _on_schema_combo_changed(self, sender, app_data) -> None:
        """Handle schema combo change."""
        self.load_schema(app_data)
    
    def _on_name_changed(self, sender, app_data) -> None:
        """Handle schema name change."""
        pass
    
    def load_schema(self, name: str) -> None:
        """Load a schema by name."""
        schema = self.schema_manager.get_schema(name)
        if not schema:
            return
        
        self.current_schema = schema
        
        # Update UI
        dpg.set_value(self.schema_name_tag, schema.name)
        dpg.set_value(self.schema_desc_tag, schema.description)
        
        # Clear and rebuild column rows
        self._clear_column_rows()
        
        for col in schema.columns:
            self._add_column_row(default_values={
                "name": col.name,
                "type": col.type.value,
                "source": col.source,
                "transform": col.transform.value,
                "required": col.required,
                "stat_test_source_column": col.stat_test_source_column,
                "stat_test_alpha": col.stat_test_alpha if col.stat_test_alpha is not None else 0.05
            })
        
        # Trigger callback
        if self.on_schema_changed:
            self.on_schema_changed(schema)
    
    def new_schema(self) -> None:
        """Create a new blank schema."""
        self.current_schema = None
        
        dpg.set_value(self.schema_name_tag, "New Schema")
        dpg.set_value(self.schema_desc_tag, "")
        
        self._clear_column_rows()
        
        # Add default columns for IPA-like structure
        self._add_column_row(default_values={
            "name": "GeneSymbol",
            "type": "string",
            "source": "auto",
            "transform": "none",
            "required": True
        })
        self._add_column_row(default_values={
            "name": "FoldChange",
            "type": "float",
            "source": "auto",
            "transform": "none",
            "required": True
        })
        self._add_column_row(default_values={
            "name": "PValue",
            "type": "float",
            "source": "auto",
            "transform": "none",
            "required": True
        })
    
    def save_schema(self) -> None:
        """Save the current schema."""
        SM, S, CD, CT, TT = get_schema_classes()
        
        name = dpg.get_value(self.schema_name_tag)
        description = dpg.get_value(self.schema_desc_tag)
        
        if not name:
            self._show_error("Please enter a schema name")
            return
        
        # Build column definitions
        columns = []
        for row in self.column_rows:
            transform_value = dpg.get_value(row["transform"])
            # Get statistical test parameters if it's a statistical test transform
            stat_test_source = None
            stat_test_method = None
            stat_test_alpha = None
            if transform_value in self.STAT_TRANSFORMS:
                stat_test_source = dpg.get_value(row["stat_source_combo"])
                stat_test_alpha = dpg.get_value(row["stat_alpha_input"])
                # Map transform to method name
                method_map = {
                    "fdr_benjamini_hochberg": "fdr_bh",
                    "fdr_benjamini_yekutieli": "fdr_by",
                    "bonferroni": "bonferroni",
                    "holm_bonferroni": "holm"
                }
                stat_test_method = method_map.get(transform_value)
            
            col = CD(
                name=dpg.get_value(row["name"]),
                type=CT(dpg.get_value(row["type"])),
                source=dpg.get_value(row["source"]),
                transform=TT(transform_value),
                required=dpg.get_value(row["required"]),
                stat_test_source_column=stat_test_source if stat_test_source else None,
                stat_test_method=stat_test_method,
                stat_test_alpha=stat_test_alpha if stat_test_alpha is not None else 0.05
            )
            columns.append(col)
        
        schema = S(
            name=name,
            description=description,
            columns=columns
        )
        
        if self.schema_manager.save_schema(schema):
            self.current_schema = schema
            
            # Update combo
            schemas = self.schema_manager.list_schemas()
            dpg.configure_item(self.schema_combo_tag, items=schemas)
            dpg.set_value(self.schema_combo_tag, name)
            
            self._show_success(f"Schema '{name}' saved successfully")
            
            if self.on_schema_changed:
                self.on_schema_changed(schema)
        else:
            self._show_error("Failed to save schema")
    
    def _delete_schema(self) -> None:
        """Delete the current schema."""
        if not self.current_schema:
            return
        
        if self.current_schema.builtin:
            self._show_error("Cannot delete built-in schemas")
            return
        
        name = self.current_schema.name
        
        if self.schema_manager.delete_schema(name):
            # Update combo
            schemas = self.schema_manager.list_schemas()
            dpg.configure_item(self.schema_combo_tag, items=schemas)
            
            if schemas:
                self.load_schema(schemas[0])
            else:
                self.new_schema()
            
            self._show_success(f"Schema '{name}' deleted")
        else:
            self._show_error("Failed to delete schema")
    
    def _apply_schema(self) -> None:
        """Apply the current schema configuration."""
        SM, S, CD, CT, TT = get_schema_classes()
        
        name = dpg.get_value(self.schema_name_tag)
        description = dpg.get_value(self.schema_desc_tag)
        
        columns = []
        for row in self.column_rows:
            transform_value = dpg.get_value(row["transform"])
            # Get statistical test parameters if it's a statistical test transform
            stat_test_source = None
            stat_test_method = None
            stat_test_alpha = None
            if transform_value in self.STAT_TRANSFORMS:
                stat_test_source = dpg.get_value(row["stat_source_combo"])
                stat_test_alpha = dpg.get_value(row["stat_alpha_input"])
                # Map transform to method name
                method_map = {
                    "fdr_benjamini_hochberg": "fdr_bh",
                    "fdr_benjamini_yekutieli": "fdr_by",
                    "bonferroni": "bonferroni",
                    "holm_bonferroni": "holm"
                }
                stat_test_method = method_map.get(transform_value)
            
            col = CD(
                name=dpg.get_value(row["name"]),
                type=CT(dpg.get_value(row["type"])),
                source=dpg.get_value(row["source"]),
                transform=TT(transform_value),
                required=dpg.get_value(row["required"]),
                stat_test_source_column=stat_test_source if stat_test_source else None,
                stat_test_method=stat_test_method,
                stat_test_alpha=stat_test_alpha if stat_test_alpha is not None else 0.05
            )
            columns.append(col)
        
        schema = S(
            name=name,
            description=description,
            columns=columns
        )
        
        self.current_schema = schema
        
        if self.on_schema_changed:
            self.on_schema_changed(schema)
        
        self._show_success("Schema applied")
    
    def show_load_dialog(self) -> None:
        """Show load schema dialog."""
        # Refresh and show combo dropdown
        schemas = self.schema_manager.list_schemas()
        dpg.configure_item(self.schema_combo_tag, items=schemas)
    
    def _show_error(self, message: str) -> None:
        """Show error popup."""
        with dpg.window(
            label="Error",
            modal=True,
            width=300,
            height=100,
            pos=[550, 400],
            on_close=lambda: dpg.delete_item("error_popup")
        ) as popup:
            dpg.set_item_alias(popup, "error_popup")
            dpg.add_text(message, wrap=-1, color=(231, 76, 60))
            dpg.add_spacer(height=10)
            dpg.add_button(
                label="OK",
                callback=lambda: dpg.delete_item("error_popup"),
                width=-1
            )
    
    def _show_success(self, message: str) -> None:
        """Show success popup."""
        with dpg.window(
            label="Success",
            modal=True,
            width=300,
            height=100,
            pos=[550, 400],
            on_close=lambda: dpg.delete_item("success_popup")
        ) as popup:
            dpg.set_item_alias(popup, "success_popup")
            dpg.add_text(message, wrap=-1, color=(46, 204, 113))
            dpg.add_spacer(height=10)
            dpg.add_button(
                label="OK",
                callback=lambda: dpg.delete_item("success_popup"),
                width=-1
            )
    
    def get_current_schema(self):
        """Get the current schema."""
        return self.current_schema
