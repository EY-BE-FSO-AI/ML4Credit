# xlsx names of regulatory template reporting docs #
    # 'LEICode_PD_ModelID_EndOfObservationPeriod_versionNumber.xlsx'
    # 'LEICode_LGD_ModelID_EndOfObservationPeriod_versionNumber.xlsx'
    # 'LEICode_CCF_ModelID_EndOfObservationPeriod_versionNumber.xlsx'
    # 'LEICode_ELBE_ModelID_EndOfObservationPeriod_versionNumber.xlsx'
    # 'LEICode_LGDD_ModelID_EndOfObservationPeriod_versionNumber.xlsx'
    # 'LEICode_SL_ModelID_EndOfObservationPeriod_versionNumber.xlsx'

#to do: grab the sheet names an loop through the sheets, mind the structure of most files is similat but not the same

class export(object):
     
     def file_path(self, target_filename):   
         import os
         
         local_dr = os.path.normpath(os.path.expanduser("~/Documents/Python"))
         path = local_dr + "/" + target_filename
         return path
    
     def xlsx(self, file, sheet, row, column, value):
          import openpyxl

          oxl = openpyxl.load_workbook(file)
          wbk = oxl.get_sheet_by_name(sheet)
          wbk.cell(row, column).value = value
          oxl.save(file)
          oxl.close
          return


import random

PD_sheet_names = {'1.0', '1.1', '1.2', '2.0', '3.0', '4.0', '5.1', '5.2'}
percent_transition_matrix = 


from export import * 

output_example = random.random()
sheet_name = '5.2'
row = 7 
col = 4 
export().xlsx(export().file_path('LEICode_PD_ModelID_EndOfObservationPeriod_versionNumber.xlsx'), sheet_name, row, col, output_example)

