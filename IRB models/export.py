# xlsx names of regulatory template reporting docs #
    # 'LEICode_PD_ModelID_EndOfObservationPeriod_versionNumber.xlsx'
    # 'LEICode_LGD_ModelID_EndOfObservationPeriod_versionNumber.xlsx'
    # 'LEICode_CCF_ModelID_EndOfObservationPeriod_versionNumber.xlsx'
    # 'LEICode_ELBE_ModelID_EndOfObservationPeriod_versionNumber.xlsx'
    # 'LEICode_LGDD_ModelID_EndOfObservationPeriod_versionNumber.xlsx'
    # 'LEICode_SL_ModelID_EndOfObservationPeriod_versionNumber.xlsx'

#to do: grab the sheet names an loop through the sheets, mind the structure of most files is similat but not the same

# class export(object):
#
#      def file_path(self, target_filename):
#          import os
#
#          local_dr = os.path.normpath(os.path.expanduser("~/Documents/Python"))
#          path = local_dr + "/" + target_filename
#          return path
#
#      def xlsx(self, file, sheet, row, column, value):
#           import openpyxl
#
#           oxl = openpyxl.load_workbook(file)
#           wbk = oxl.get_sheet_by_name(sheet)
#           wbk.cell(row, column).value = value
#           oxl.save(file)
#           oxl.close
#           return
#

# import random
#
# PD_sheet_names = {'1.0', '1.1', '1.2', '2.0', '3.0', '4.0', '5.1', '5.2'}
#
#
# from export import *
#
# output_example = random.random()
# sheet_name = '5.2'
# row = 7
# col = 4
# export().xlsx(export().file_path('LEICode_PD_ModelID_EndOfObservationPeriod_versionNumber.xlsx'), sheet_name, row, col, output_example)

import openpyxl
import pandas as pd

class export(object):

    def file_path(self, target_filename):
        import os

        local_dr = os.path.normpath(os.path.expanduser("~/Documents/Python"))
        path = local_dr + "/" + target_filename
        return path

    def open_wb(self, file_name):
        return openpyxl.load_workbook(file_name)

    def save_wb(self, file_name):
        return openpyxl.save(file_name)

    def array_toExcel(self, wb, stat_array, row_pos, col_pos, row_wise=True):
        """
        Write array to excel.
        :param wb: excel openpyxl workbook;
        :param stat_array: array to write to excel;
        :param row_pos: initial row positions;
        :param col_pos: column position;
        :param row_wise: True/False.
        :return:
        """
        i = 0
        for stat in stat_array:
            if row_wise:
                wb.cell(row= row_pos + i, column= col_pos).value = stat
                i += 1
            else:
                wb.cell(row= row_pos, column= col_pos + i).value = stat
                i += 1
        return None

    def value_toExcel(self, file, sheet, row, column, value):

        oxl = openpyxl.load_workbook(file)
        wbk = oxl.get_sheet_by_name(sheet)
        wbk.cell(row, column).value = value
        oxl.save(file)
        oxl.close
        return None

    def PD_toExcel(self, pd_inputs):
        """
        Fill PD test statistics and other information to Excel file;
        :param pd_inputs: dictionary containing pd test results, details etc.
        :return: Save to excel results.
        """
        file_name = self.file_path(target_filename="LEICode_PD_ModelID_EndOfObservationPeriod_versionNumber.xlsx")
        oxl = self.open_wb(file_name)

        # Predictive ability
        # PD Back-testing using a Jeffreys test (§ 2.5.3.1) - sheet 3.0
        wbk30 = oxl.get_sheet_by_name("3.0")
        self.array_toExcel(wb=wbk30, stat_array=pd_inputs["predictive_ability"][0],row_pos=8, col_pos=4) # Rating grade names
        self.array_toExcel(wb=wbk30, stat_array=pd_inputs["predictive_ability"][1], row_pos=8, col_pos=5) # Average PD
        self.array_toExcel(wb=wbk30, stat_array=pd_inputs["predictive_ability"][2], row_pos=8, col_pos=6) # Nb of customers
        self.array_toExcel(wb=wbk30, stat_array=pd_inputs["predictive_ability"][3], row_pos=8, col_pos=7) # Nb of defaults
        self.array_toExcel(wb=wbk30, stat_array=pd_inputs["predictive_ability"][4], row_pos=8, col_pos=8) # Jeffrey p-vals
        self.array_toExcel(wb=wbk30, stat_array=pd_inputs["predictive_ability"][5], row_pos=8, col_pos=9) # Original exposure

        # Discriminatory Power
        # Current AUC vs AUC at initial validation/development (§ 2.5.4.1) - sheet 4.0
        wbk40 = oxl.get_sheet_by_name("4.0")
        self.array_toExcel(wb=wbk40, stat_array=pd_inputs["AUC"], row_pos=7, col_pos=4, row_wise=False)

        # Stability - Customer Migration
        # Current AUC vs AUC at initial validation/development (§ 2.5.4.1) - sheet 4.0
        self.array_toExcel(wb=wbk40, stat_array=pd_inputs["concentration_rating_grades"], row_pos=18, col_pos=4,
                           row_wise=False)
        # Customer Migrations (§ 2.5.5.1) - sheet 5.1
        wbk51 = oxl.get_sheet_by_name("5.1")
        self.array_toExcel(wb=wbk51, stat_array=pd_inputs["customer_migrations"], row_pos=7, col_pos=4,row_wise=False)

        # Stability - Stability of Migrations
        # Customer Migrations (§ 2.5.5.2) - sheet 5.2
        wbk52 = oxl.get_sheet_by_name("5.2")
        transMatrix = pd_inputs["stability_migration_matrix"][0]
        z_ij = pd_inputs["stability_migration_matrix"][1] + pd_inputs["stability_migration_matrix"][2]
        phi_zij = pd_inputs["stability_migration_matrix"][3] + pd_inputs["stability_migration_matrix"][4]
        c = 0
        for j in range(len(transMatrix.columns)):
            self.array_toExcel(wb=wbk52, stat_array= transMatrix.iloc[:, j], row_pos=7, col_pos=(4 + c))
            self.array_toExcel(wb=wbk52, stat_array= z_ij[:, j], row_pos=7, col_pos=(5 + c))
            self.array_toExcel(wb=wbk52, stat_array= phi_zij[:, j], row_pos=7, col_pos=(6 + c))
            c += 3

        # Save file
        oxl.save( file_name )
        oxl.close()
        return "PD results saved to Excel."

    def LGD_toExcel(self):
        file_name = self.file_path(target_filename="LEICode_PD_ModelID_EndOfObservationPeriod_versionNumber.xlsx")
        oxl = openpyxl.load_workbook(file_name)

        # Save file
        oxl.save(file_name)
        oxl.close()
        return "LGD results saved to Excel."
