# field_performance_visualizer.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, Any, List, Tuple
from loguru import logger
from pathlib import Path
import xlsxwriter # NEW: Import for Excel writing

# IMPORTANT: These lists should match the definitions in your evaluator.py
# Ensure consistency if you modify them in evaluator.py
COMPLETENESS_CHECK_FIELDS = [
    "event_type", "event_sub_type", "state_of_victim", "victim_gender",
    "specified_matter", "date_reference", "frequency", "repeat_incident",
    "identification", "injury_type", "victim_age", "victim_relation",
    "incident_location", "area", "suspect_description", "object_involved",
    "date_of_birth", "used_weapons", "offender_relation", "mode_of_threat",
    "need_ambulance", "children_involved", "generated_event_sub_type_detail"
]

# Define the fields that are strictly categorical (used for direct equality check)
# These are the fields where `_strict_match` or `_accuracy` is directly used.
CATEGORICAL_FIELDS_FOR_STRICT_MATCH = [
    'event_type', 'event_sub_type', 'state_of_victim', 'victim_gender',
    'need_ambulance', 'children_involved', 'repeat_incident'
]

# Define the text-based fields that use LLM binary similarity for correctness
TEXT_FIELDS_FOR_LLM_BINARY_SIMILARITY = [
    "specified_matter", "date_reference", "frequency",
    "identification", "injury_type", "victim_age", "victim_relation",
    "incident_location", "area", "suspect_description", "object_involved",
    "date_of_birth", "used_weapons", "offender_relation", "mode_of_threat",
    "generated_event_sub_type_detail"
]


class FieldPerformanceVisualizer:
    def __init__(self, output_dir: str = "visualizations/field_performance_matrix"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Field performance visualizations will be saved to: {self.output_dir}")

        # Define colors for the heatmap and Excel output
        self.colors = {
            "Correct": "#28a745",   # Correct (GT and LLM, and correct)
            "Incorrect": "#fd7e14",  # Incorrect (GT and LLM, but wrong)
            "Missing": "#ffc107",  # Missing (GT present, LLM absent)
            "Hallucination": "#dc3545"      # Hallucination (GT absent, LLM present)
        }
        self.color_map = lis t(self.colors.values())
        self.color_labels = list(self.colors.keys())

    def _get_field_status(self, field_name: str, record_metrics: Dict[str, Any]) -> str:
        """
        Determines the status of a single field for a single record based on evaluation metrics.
        This function relies on the detailed_results structure produced by evaluator.py.
        """
        # Retrieve hallucination and missing lists for direct checks
        hallucinated_list = record_metrics.get('hallucinated_fields_list', [])
        missing_list = record_metrics.get('missing_from_llm_list', [])

        if field_name in hallucinated_list:
            return "Red" # Hallucination: LLM present, GT absent
        
        if field_name in missing_list:
            return "Yellow" # Missing: GT present, LLM absent

        # If not hallucinated or missing, it implies both GT and LLM had a value (or both were 'not specified')
        # We now need to check if that value was correct or incorrect.

        if field_name in CATEGORICAL_FIELDS_FOR_STRICT_MATCH:
            # For categorical fields, check the '{field_name}_strict_match' metric
            # This aligns with how evaluator.py stores the accuracy for these fields in detailed_results
            accuracy = record_metrics.get(f'{field_name}_strict_match')
            if accuracy is not None:
                return "Green" if accuracy == 1.0 else "Orange"
            else:
                # This case implies the field was not processed or metrics are missing,
                # which shouldn't happen if evaluator.py is working correctly.
                logger.warning(f"Categorical field '{field_name}' had no accuracy metric. Defaulting to Neutral.")
                return "Neutral" # Should ideally not be reached
        
        elif field_name in TEXT_FIELDS_FOR_LLM_BINARY_SIMILARITY:
            # For text fields, check the '{field_name}_llm_binary_similarity' metric
            # This aligns with how evaluator.py stores the binary similarity for these fields in detailed_results
            llm_binary_sim = record_metrics.get(f'{field_name}_llm_binary_similarity')
            if llm_binary_sim is not None:
                return "Green" if llm_binary_sim == 1.0 else "Orange"
            else:
                logger.warning(f"Text field '{field_name}' had no llm_binary_similarity metric. Defaulting to Neutral.")
                return "Neutral" # Should ideally not be reached
        
        else:
            # Fallback for any field not explicitly categorized (e.g., if COMPLETENESS_CHECK_FIELDS
            # contains fields not in either CATEGORICAL_FIELDS_FOR_STRICT_MATCH or TEXT_FIELDS_FOR_LLM_BINARY_SIMILARITY)
            logger.warning(f"Field '{field_name}' not found in known categorical or text fields for status determination. Defaulting to Neutral.")
            return "Neutral" # This means the field was not handled by the logic above.

    def generate_field_performance_matrix(self, evaluation_results: Dict[str, Any], output_filename: str = "field_performance_matrix_heatmap.png"):
        """
        Generates a heatmap visualizing the performance of each field across all text files.
        Colors: Green (Correct), Orange (Incorrect), Yellow (Missing), Red (Hallucination).
        Also prints counts for each category.
        """
        detailed_results = evaluation_results.get('detailed_results', [])
        if not detailed_results:
            logger.warning("No detailed results found for field performance matrix. Skipping plot.")
            return

        # Extract file names and field names
        file_names = [os.path.basename(d['file_name']) for d in detailed_results]
        field_names = COMPLETENESS_CHECK_FIELDS # Use the comprehensive list of fields

        # Create an empty matrix to store numerical indices for colors
        color_to_index = {label: i for i, label in enumerate(self.color_labels)}
        # Initialize with -1, which will not be mapped by the colormap, effectively being transparent if not assigned
        matrix = np.full((len(field_names), len(file_names)), -1, dtype=int)

        # Counters for each category
        category_counts = {label: 0 for label in self.color_labels}

        for file_idx, record_metrics in enumerate(detailed_results):
            for field_idx, field_name in enumerate(field_names):
                status = self._get_field_status(field_name, record_metrics)
                
                # Assign status to matrix and increment counter
                if status in color_to_index:
                    matrix[field_idx, file_idx] = color_to_index[status]
                    category_counts[status] += 1
                else:
                    logger.warning(f"Unknown status '{status}' for field '{field_name}' in file '{file_names[file_idx]}'. Skipping.")

        # Create a DataFrame for easier plotting with seaborn
        df_matrix = pd.DataFrame(matrix, index=field_names, columns=file_names)

        # Filter out rows/columns that might be entirely -1 (unassigned)
        # This ensures the plot only shows fields/files that actually had a status determined.
        df_matrix = df_matrix.loc[(df_matrix != -1).any(axis=1), (df_matrix != -1).any(axis=0)]
        
        if df_matrix.empty:
            logger.warning("Filtered DataFrame for heatmap is empty. No data to plot.")
            return

        # Plotting the heatmap
        # Adjust figure size dynamically based on number of files and fields
        fig_width = max(15, len(df_matrix.columns) * 0.5) # Minimum 15, then scale by files
        fig_height = max(10, len(df_matrix.index) * 0.3) # Minimum 10, then scale by fields
        plt.figure(figsize=(fig_width, fig_height))
        
        # Create a custom colormap from the defined colors
        cmap = plt.cm.colors.ListedColormap(self.color_map)
        
        # Adjust bounds for the colormap to match the number of categories
        # Each category gets a unique integer from 0 to len(color_labels)-1
        bounds = np.arange(len(self.color_labels) + 1) - 0.5
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

        ax = sns.heatmap(df_matrix, cmap=cmap, norm=norm,
                         linewidths=.5, linecolor='lightgray',
                         cbar_kws={"ticks": np.arange(len(self.color_labels)), "label": "Status"})

        # Manually set colorbar labels
        cbar = ax.collections[0].colorbar
        cbar.set_ticks(np.arange(len(self.color_labels)))
        cbar.set_ticklabels(self.color_labels)

        plt.title('Field Performance Matrix: LLM Output vs Ground Truth', fontsize=16)
        plt.xlabel('Text File', fontsize=12)
        plt.ylabel('Field', fontsize=12)
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        
        save_path = self.output_dir / output_filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Field performance matrix plot saved to {save_path}")

        # Print exact counts
        logger.info("\n--- Field Performance Category Counts ---")
        total_cells_processed = sum(category_counts.values())
        if total_cells_processed > 0:
            for category, count in category_counts.items():
                percentage = (count / total_cells_processed) * 100
                logger.info(f"{category}: {count} / {total_cells_processed} ({percentage:.2f}%)")
        else:
            logger.info("No fields were processed to determine status categories.")
        logger.info("---------------------------------------")

    def generate_color_coded_excel(self, evaluation_results: Dict[str, Any], predictions: List[Any], ground_truth_map: Dict[str, Any], output_excel_filename: str = "field_performance_summary.xlsx"):
        """
        Generates an Excel file with two sheets:
        1. 'LLM Performance Matrix': LLM values with color coding based on status.
        2. 'Ground Truth Values': Ground truth values for reference.
        """
        detailed_results = evaluation_results.get('detailed_results', [])
        if not detailed_results:
            logger.warning("No detailed results found for Excel generation. Skipping.")
            return

        # Ensure output directory exists
        excel_path = self.output_dir / output_excel_filename
        logger.info(f"Generating color-coded Excel file to: {excel_path}")

        # Create a new Excel workbook and add a worksheet.
        workbook = xlsxwriter.Workbook(excel_path)

        # Define formats for colors
        formats = {}
        for label, hex_color in self.colors.items():
            formats[label] = workbook.add_format({'bg_color': hex_color, 'border': 1})
        # Add a default format for unassigned/neutral cells if needed
        formats["Neutral"] = workbook.add_format({'border': 1})
        formats["Header"] = workbook.add_format({'bold': True, 'bg_color': '#D3D3D3', 'border': 1, 'align': 'center', 'valign': 'vcenter'})
        formats["Filename_Header"] = workbook.add_format({'bold': True, 'bg_color': '#D3D3D3', 'border': 1, 'align': 'center', 'valign': 'vcenter', 'rotation': 90})


        # --- Sheet 1: LLM Performance Matrix (with color coding) ---
        llm_sheet = workbook.add_worksheet('LLM Performance Matrix')
        
        # Write column headers (file names)
        file_names_clean = [os.path.basename(d['file_name']) for d in detailed_results]
        llm_sheet.write(0, 0, "Field Name", formats["Header"]) # Top-left corner
        for col_idx, file_name in enumerate(file_names_clean):
            llm_sheet.write(0, col_idx + 1, file_name, formats["Filename_Header"])
            llm_sheet.set_column(col_idx + 1, col_idx + 1, 15) # Set column width

        # Write row headers (field names) and data with colors
        for row_idx, field_name in enumerate(COMPLETENESS_CHECK_FIELDS):
            llm_sheet.write(row_idx + 1, 0, field_name, formats["Header"]) # Field name as row header
            llm_sheet.set_row(row_idx + 1, 40) # Set row height for readability

            for col_idx, record_metrics in enumerate(detailed_results):
                file_name_key = os.path.basename(record_metrics['file_name'])
                
                # Find the corresponding prediction object
                pred_obj = next((p for p in predictions if os.path.basename(p.file_name) == file_name_key), None)
                
                llm_value = "N/A (No Prediction)"
                if pred_obj:
                    # Get the actual value from the ProcessedOutput object
                    llm_value_raw = getattr(pred_obj, field_name, None)
                    llm_value = str(llm_value_raw) if llm_value_raw is not None and str(llm_value_raw).strip().lower() not in ["null", "none", "not specified", ""] else ""

                status = self._get_field_status(field_name, record_metrics)
                cell_format = formats.get(status, formats["Neutral"]) # Get format based on status

                llm_sheet.write(row_idx + 1, col_idx + 1, llm_value, cell_format)
        
        llm_sheet.set_column(0, 0, 25) # Set width for field name column


        # --- Sheet 2: Ground Truth Values (for reference) ---
        gt_sheet = workbook.add_worksheet('Ground Truth Values')

        # Write column headers (file names)
        gt_sheet.write(0, 0, "Field Name", formats["Header"])
        for col_idx, file_name in enumerate(file_names_clean):
            gt_sheet.write(0, col_idx + 1, file_name, formats["Filename_Header"])
            gt_sheet.set_column(col_idx + 1, col_idx + 1, 15) # Set column width

        # Write row headers (field names) and ground truth data
        for row_idx, field_name in enumerate(COMPLETENESS_CHECK_FIELDS):
            gt_sheet.write(row_idx + 1, 0, field_name, formats["Header"])
            gt_sheet.set_row(row_idx + 1, 40) # Set row height

            for col_idx, file_name in enumerate(file_names_clean):
                gt_obj = ground_truth_map.get(file_name) # Get GT object from the map
                
                gt_value = "N/A (No Ground Truth)"
                if gt_obj:
                    # Get the actual value from the GroundTruthOutput object
                    gt_value_raw = getattr(gt_obj, field_name, None)
                    gt_value = str(gt_value_raw) if gt_value_raw is not None and str(gt_value_raw).strip().lower() not in ["null", "none", "not specified", ""] else ""
                
                gt_sheet.write(row_idx + 1, col_idx + 1, gt_value, formats["Neutral"]) # No special coloring for GT

        gt_sheet.set_column(0, 0, 25) # Set width for field name column

        # Close the workbook
        workbook.close()
        logger.info(f"Excel file generated successfully at {excel_path}")
