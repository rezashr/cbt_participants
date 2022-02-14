from gspread_formatting import *

# Minimum gap between consecutive CBT sessions.
min_gap_between_sessions = 9

# Number of days to report a participant as eligible before she/he is due for their CBT session.
N_DAYS_BEFORE_TO_REPORT = 14

# Column names in the spreadsheet.
date_col_names_per_year = ('Baseline', 'Yr 1', 'Yr 2', 'Yr 3')
acceptable_intervals_months = ((-1, 6), (11, 18), (23, 30), (35, 48))

# Some formatting things. Apparently people like pretty things!!!
grace_period_format = cellFormat(backgroundColor=color(0.5, 0.8, 1))
critical_booking_format = cellFormat(backgroundColor=color(1, 0.5, 0.5))
within_interval_booking_format = cellFormat(backgroundColor=color(1, 0.85, 0.85))
future_booking_format = cellFormat(backgroundColor=color(0.5, 1, 0.5))

EEG_MRI_NP_SPREADSHEET = 'MRI_EEG_NP_sessions'
MRI_WORKSHEET = 'MRI Sessions'
EEG_WORKSHEET = 'EEG Sessions'

CBT_ELIGIBLE_PARTICIPANTS_SPREADSHEET = 'List of eligible participants for CT'
ELIGIBLE_WORKSHEET = 'Eligible_new'
DUPLICATE_WORKSHEET = 'Potentially duplicate records'

date_format_str = '%Y-%m-%d'

EEG_date_column = 'EEG_date'
MRI_date_column = 'scan_date'