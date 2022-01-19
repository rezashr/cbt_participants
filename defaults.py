from gspread_formatting import *

# Number of months between NP date and the CBT.
ACCEPTABLE_GAP_months = [1, 6]  # [Before, After]
ACCEPTABLE_GAP_months_end_of_study = [1, 12]

# Number of months of grace to attempt to collect CBT.
Months_of_grace = 2

# Minimum gap between consecutive CBT sessions.
min_gap_between_sessions = 9

# Number of days to report a participant as eligible before she/he is due for their CBT session.
N_DAYS_BEFORE_TO_REPORT = 14

# Column names in the spreadsheet.
date_col_names_per_year = ('Baseline', 'Yr 1', 'Yr 2', 'Yr 3')

# Some formatting things. Apparently people like pretty things!!!
grace_period_format = cellFormat(backgroundColor=color(0.5, 0.8, 1))
critical_booking_format = cellFormat(backgroundColor=color(1, 0.5, 0.5))
within_interval_booking_format = cellFormat(backgroundColor=color(1, 0.85, 0.85))
future_booking_format = cellFormat(backgroundColor=color(0.5, 1, 0.5))
