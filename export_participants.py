import pandas as pd
import gspread
from pathlib import Path
from gspread_dataframe import get_as_dataframe, set_with_dataframe
from pychchpd import pychchpd
from datetime import datetime
from dateutil import relativedelta
import numpy as np
from defaults import *


def apply_formatting(worksheet, rows, fmt):
    if len(rows) > 0:
        start_row = rows[0] + 2
        end_row = rows[-1] + 2
        if start_row == end_row:
            fmt_rows = start_row.__str__()
        else:
            fmt_rows = f'{start_row}:{end_row}'

        format_cell_ranges(worksheet, [(fmt_rows, fmt)])


def run():
    # Retrieve participant names.
    chchpd = pychchpd.chchpd(use_server=True)
    chchpd.anonymize(False)
    participants = (chchpd.import_participants(identifiers=True).
                    select_columns('first_name', 'last_name', 'subject_id').
                    assign(Fullname=lambda x: x['first_name'] + " " + x['last_name']).
                    drop(columns=['first_name', 'last_name']))

    # Authenticate google account.
    oauth_file = Path(__file__).parent.joinpath('oauth.json').__str__()
    gc = gspread.oauth(credentials_filename=oauth_file)

    # Read current MRI spreadsheet (exported from R).
    mri_data_ws = gc.open('MRI_EEG_NP_sessions').worksheet('MRI Sessions')
    mri_data = get_as_dataframe(mri_data_ws, evaluate_formulas=True, skipinitialspace=True,
                                parse_dates=True, keep_date_col=True, infer_datetime_format=False)

    # Remove participants who did not get a scan.
    mri_eligible_participants = (mri_data[mri_data.scan_number.notna()].
                                 select_columns('subject_id', 'session_date', 'diagnosis').
                                 merge(participants, how='left'))

    # Get current list of eligible participants.
    eligible_ss = gc.open('List of eligible participants for CT')
    eligible_ws = eligible_ss.worksheet('Eligible_new')
    current_eligible_participants = get_as_dataframe(eligible_ws)

    # Remove empty rows (google spreadsheet does this...)
    current_eligible_participants = current_eligible_participants[current_eligible_participants.notna().any(axis=1)]

    if 'Fullname' in current_eligible_participants.columns:
        current_eligible_participants = current_eligible_participants.drop(columns=['Fullname'])

    # Add fullname.
    current_eligible_participants = current_eligible_participants.merge(participants, how='left')
    cols = current_eligible_participants.columns.tolist()
    cols.remove('Fullname')
    cols = ['Fullname'] + cols

    current_eligible_participants = current_eligible_participants[cols]

    new_participants = (current_eligible_participants.
                        merge(mri_eligible_participants, how='right', indicator=True, copy=True).
                        query('_merge == "right_only"').
                        select_columns(mri_eligible_participants.columns))

    if not new_participants.empty:
        current_eligible_participants = pd.concat((current_eligible_participants, new_participants), ignore_index=True)

    set_with_dataframe(worksheet=eligible_ws, dataframe=current_eligible_participants,
                       include_index=False, include_column_header=True, allow_formulas=True,
                       resize=True)

    participants_to_book = pd.DataFrame()
    participants_to_check = pd.DataFrame()

    today = datetime.today().date()
    for idx, row in current_eligible_participants.iterrows():
        if pd.isna(row.session_date):
            continue
        session_date = pd.to_datetime(row.session_date, utc=False).date()
        for i in range(0, 4):
            visit_date = session_date + relativedelta.relativedelta(years=i)
            if i < 3:
                report_after = visit_date - relativedelta.relativedelta(months=ACCEPTABLE_GAP_months[0],
                                                                        days=N_DAYS_BEFORE_TO_REPORT)

                report_before = visit_date + relativedelta.relativedelta(months=ACCEPTABLE_GAP_months[1])

            else:
                report_after = visit_date - relativedelta.relativedelta(months=ACCEPTABLE_GAP_months_end_of_study[0],
                                                                        days=N_DAYS_BEFORE_TO_REPORT)

                report_before = visit_date + relativedelta.relativedelta(months=ACCEPTABLE_GAP_months_end_of_study[1])

            previous_visit = None
            if i > 0:  # After baseline
                try:
                    previous_visit = pd.to_datetime(row[date_col_names_per_year[i - 1]], utc=False).date()
                    if pd.isna(previous_visit):
                        previous_visit = None

                except Exception:
                    pass

            if previous_visit:
                report_after = max(report_after,
                                   previous_visit + relativedelta.relativedelta(months=min_gap_between_sessions))

            report_grace = report_before + relativedelta.relativedelta(months=Months_of_grace)

            if report_after >= report_before:
                new_row = row.copy()
                new_row[date_col_names_per_year[i]] = 'Oh noo, please give me some attention...'
                participants_to_check = participants_to_check.append(new_row)

            if report_after <= today <= report_grace:
                if pd.isna(row[date_col_names_per_year[i]]) or row[date_col_names_per_year[i]].strip == '':
                    new_row = row[['Fullname', 'subject_id', 'diagnosis', 'session_date']].copy()

                    if i < 3:
                        collect_after = visit_date - relativedelta.relativedelta(months=ACCEPTABLE_GAP_months[0])
                        collect_before = visit_date + relativedelta.relativedelta(months=ACCEPTABLE_GAP_months[1])

                    # This would be the end of study.
                    else:
                        collect_after = visit_date - relativedelta.relativedelta(
                            months=ACCEPTABLE_GAP_months_end_of_study[0])

                        collect_before = visit_date + relativedelta.relativedelta(
                            months=ACCEPTABLE_GAP_months_end_of_study[1])

                    if previous_visit:
                        collect_after = max(collect_after, previous_visit + relativedelta.relativedelta(
                            months=min_gap_between_sessions))

                    new_row['Session to collect'] = date_col_names_per_year[i].replace('date', '').strip()
                    new_row['To collect after'] = collect_after
                    new_row['To collect before'] = collect_before
                    participants_to_book = participants_to_book.append(new_row, ignore_index=True)

    if not participants_to_book.empty:
        participants_to_book = participants_to_book.sort_values(by='To collect before', ignore_index=True,
                                                                ascending=True)

    participants_to_book_ws = eligible_ss.worksheet('Participants to book')
    participants_to_book_ws.clear()
    set_with_dataframe(worksheet=participants_to_book_ws, dataframe=participants_to_book,
                       include_index=False, include_column_header=True, allow_formulas=True,
                       resize=True)

    set_frozen(participants_to_book_ws, rows=1)

    interval = np.array(
        [x.total_seconds() / 24 / 3600 for x in (participants_to_book['To collect before'] - today).tolist()])

    grace_rows = np.where(interval <= 0)[0]
    critical_rows = np.where((0 < interval) & (interval <= 28))[0]
    within_interval_rows = np.where((interval >= 28) & (today > participants_to_book['To collect after']))[0]
    future_rows = np.where(today < participants_to_book['To collect after'])[0]

    apply_formatting(participants_to_book_ws, rows=grace_rows, fmt=grace_period_format)
    apply_formatting(participants_to_book_ws, rows=critical_rows, fmt=critical_booking_format)
    apply_formatting(participants_to_book_ws, rows=within_interval_rows, fmt=within_interval_booking_format)
    apply_formatting(participants_to_book_ws, rows=future_rows, fmt=future_booking_format)

    if not participants_to_check.empty:
        try:
            check_ws = eligible_ss.worksheet('Participants to check')
            check_ws.clear()
        except gspread.WorksheetNotFound:
            check_ws = eligible_ss.add_worksheet('Participants to check', rows=0, cols=0)

        set_with_dataframe(check_ws, dataframe=participants_to_check, resize=True)

    else:
        try:
            eligible_ss.del_worksheet('Participants to check')
        except Exception:
            pass


if __name__ == '__main__':
    exit(run())
