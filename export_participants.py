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


def fix_date_columns(data):
    data = data.copy()
    date_columns = data.columns[data.columns.str.contains('date', case=False)]
    for col in date_columns:
        data[col] = pd.to_datetime(data[col], yearfirst=True, errors='coerce')

    return data


def df_to_strdate(data, date_format=None):
    import numpy as np
    data = data.copy()
    if date_format is None:
        date_format = date_format_str

    def _conversion(x):
        if pd.notna(x):
            return x.strftime(date_format)
        else:
            return ''

    date_columns = data.fillna(np.nan).select_dtypes(include=np.datetime64).columns
    for col in date_columns:
        data[col] = data.apply(lambda x: _conversion(x[col]), axis=1)

    return data


class ComputerizedTestsParticipants:
    def __init__(self):
        self._chchpd = None
        self._participants = None
        self._gc = None
        self._mri_data = None
        self._eeg_data = None
        self._latest_data = None

    @property
    def chchpd(self):
        if self._chchpd is None:
            self._chchpd = pychchpd.chchpd(use_server=True)
            self._chchpd.anonymize(False)
        return self._chchpd

    @property
    def participants(self):
        if self._participants is None:
            # Retrieve participant names.
            self._participants = (self.chchpd.import_participants(identifiers=True).
                                  select_columns('first_name', 'last_name', 'subject_id', 'dead').
                                  assign(Fullname=lambda x: x['first_name'] + " " + x['last_name']).
                                  drop(columns=['first_name', 'last_name']))

        return self._participants.copy()

    @property
    def gc(self, oauth_file=None):
        if self._gc is None:
            if oauth_file is None:
                # Authenticate google account.
                oauth_file = Path(__file__).parent.joinpath('oauth.json').__str__()
            self._gc = gspread.oauth(credentials_filename=oauth_file)

        return self._gc

    @property
    def mri_data(self):
        if self._mri_data is None:
            mri_data_ws = self.gc.open(EEG_MRI_NP_SPREADSHEET).worksheet(MRI_WORKSHEET)
            mri_data = get_as_dataframe(mri_data_ws, evaluate_formulas=True, skipinitialspace=True,
                                        parse_dates=True, keep_date_col=True, infer_datetime_format=False)

            mri_data = fix_date_columns(mri_data)

            # Remove participants who did not get a scan.
            mri_eligible_participants = (mri_data[mri_data.scan_number.notna()].
                                         select_columns('subject_id', 'session_date', MRI_date_column, 'diagnosis').
                                         merge(self.participants, how='left'))

            self._mri_data = mri_eligible_participants

        return self._mri_data.copy()

    @property
    def eeg_data(self):
        if self._eeg_data is None:
            # Get EEG sessions
            eeg_data_ws = self.gc.open(EEG_MRI_NP_SPREADSHEET).worksheet(EEG_WORKSHEET)
            eeg_data = get_as_dataframe(eeg_data_ws, evaluate_formulas=True, skipinitialspace=True,
                                        parse_dates=True, keep_date_col=True, infer_datetime_format=False)

            eeg_data = fix_date_columns(eeg_data)

            eeg_data['subject_id'] = eeg_data.apply(lambda x: x.EEG_session_id.rsplit('_', maxsplit=1)[0], axis=1)

            self._eeg_data = eeg_data

        return self._eeg_data.copy()

    @property
    def latest_data(self):
        if self._latest_data is None:
            data = self.mri_data.copy()
            data[EEG_date_column] = None
            for row_idx, row in data.iterrows():
                eeg_row = self.eeg_data.query(f'subject_id == "{row.subject_id}"').copy()
                if not eeg_row.empty:
                    min_idx = (eeg_row[EEG_date_column] - row[MRI_date_column]).abs().argmin()
                    data.iloc[row_idx, data.columns.get_loc(EEG_date_column)] = eeg_row[EEG_date_column].iloc[min_idx]

            data = data[
                ['Fullname', 'subject_id', 'session_date', MRI_date_column, EEG_date_column, 'diagnosis', 'dead']]

            data = data.sort_values(MRI_date_column)
            self._latest_data = data

        return self._latest_data.copy()

    def current_eligible_participants(self):
        eligible_ss = self.gc.open(CBT_ELIGIBLE_PARTICIPANTS_SPREADSHEET)
        eligible_ws = eligible_ss.worksheet(ELIGIBLE_WORKSHEET)
        current_eligible_participants = get_as_dataframe(eligible_ws)

        # Remove empty rows (google spreadsheet does this...)
        current_eligible_participants = current_eligible_participants[current_eligible_participants.notna().any(axis=1)]
        current_eligible_participants = fix_date_columns(current_eligible_participants)

        return current_eligible_participants

    def _add_delete_worksheet(self, data, ws_title, ss_title=None):
        # By default add new worksheets to the CBT spreadsheet.
        if ss_title is None:
            ss_title = CBT_ELIGIBLE_PARTICIPANTS_SPREADSHEET

        ss = self.gc.open(ss_title)

        # Add a worksheet for duplicate records.
        if not data.empty:
            try:
                ws = ss.worksheet(ws_title)
                ws.clear()
            except gspread.WorksheetNotFound:
                ws = ss.add_worksheet(ws_title, rows=0, cols=0)

            data = df_to_strdate(data.copy())
            set_with_dataframe(ws, dataframe=data, resize=True)
            set_frozen(ws, rows=1)
        else:
            # Empty dataset, so remove the worksheet if exists.
            try:
                ws = ss.worksheet(ws_title)
                ss.del_worksheet(ws)
            except Exception:
                pass

    def update_eligible_participants_list(self):
        current_eligible_participants = self.current_eligible_participants()
        latest_eligible_data = self.latest_data

        for col_name in ['Fullname', EEG_date_column, 'session_date', 'diagnosis', 'dead']:
            if col_name in current_eligible_participants.columns:
                current_eligible_participants = current_eligible_participants.drop(columns=[col_name])

        # New participants (to be added to the spreadsheet).
        new_participants = (current_eligible_participants.
                            merge(latest_eligible_data, how='right', indicator=True, copy=True).
                            query('_merge == "right_only"').
                            select_columns(current_eligible_participants.columns))

        # Duplicate or erroneous data.
        potentially_duplicate_records = (current_eligible_participants.
                                         merge(latest_eligible_data, how='left', indicator=True,
                                               copy=True).
                                         query('_merge == "left_only"').
                                         select_columns(current_eligible_participants.columns))

        self._add_delete_worksheet(potentially_duplicate_records, DUPLICATE_WORKSHEET)

        if not new_participants.empty:
            eligible_participants = pd.concat((current_eligible_participants, new_participants),
                                              ignore_index=True)
        else:
            eligible_participants = current_eligible_participants

        eligible_participants = eligible_participants.merge(latest_eligible_data, how='left')
        cols = latest_eligible_data.columns.tolist()
        for col in eligible_participants.columns.tolist():
            if col not in cols:
                cols += [col]

        eligible_participants = eligible_participants[cols]

        eligible_ws = self.gc.open(CBT_ELIGIBLE_PARTICIPANTS_SPREADSHEET).worksheet(ELIGIBLE_WORKSHEET)
        set_with_dataframe(worksheet=eligible_ws, dataframe=df_to_strdate(eligible_participants),
                           include_index=False, include_column_header=True, allow_formulas=True,
                           resize=True)

        return eligible_participants

    def _compute_interval_individual(self, row, date_column, modality=None):
        date = pd.to_datetime(row[date_column])
        today = datetime.today().date()
        if pd.isna(date):
            return None

        N_sessions = len(acceptable_intervals_months)
        interval_dict = {'session'       : [],
                         'modality'      : [],
                         'status'        : [],
                         'date_collected': [],
                         'interval_start': [],
                         'interval_end'  : [],
                         'needs_checking': []}

        for i in range(N_sessions):
            session_name = date_col_names_per_year[i]
            interval_start_min = None
            if i > 0:
                previous_date = row[date_col_names_per_year[i - 1]]
                previous_date = pd.to_datetime(previous_date, errors='coerce')
                if pd.notna(previous_date):
                    interval_start_min = previous_date + relativedelta.relativedelta(months=min_gap_between_sessions)

            interval_start = date + relativedelta.relativedelta(months=acceptable_intervals_months[i][0])
            if interval_start_min is not None:
                interval_start = max(interval_start_min, interval_start)

            interval_end = date + relativedelta.relativedelta(months=acceptable_intervals_months[i][1])

            date_collected = row[session_name]
            flagged = 'No'
            deceased = row['dead']
            if pd.isna(date_collected):
                if interval_end.date() < today:
                    status = 'Missed'
                elif deceased:
                    status = 'Deceased'
                else:
                    if interval_end.date() <= (today + relativedelta.relativedelta(days=7)):
                        status = 'Should be collected within 7 days.'
                    elif interval_end.date() <= (today + relativedelta.relativedelta(days=14)):
                        status = 'Should be collected within 14 days.'
                    elif interval_end.date() <= (today + relativedelta.relativedelta(days=30)):
                        status = 'Should be collected within 30 days.'
                    else:
                        status = 'Can be collected.'

            elif pd.isna(pd.to_datetime(date_collected, errors='coerce')):
                status = 'Missed'

            else:
                status = 'Collected'
                date_collected = pd.to_datetime(date_collected, errors='coerce')
                if (date_collected < interval_start) or (date_collected > interval_end):
                    flagged = 'Yes'
                date_collected = date_collected.date()

            interval_dict['session'].append(session_name)
            interval_dict['status'].append(status)
            interval_dict['date_collected'].append(date_collected)
            interval_dict['interval_start'].append(interval_start.date())
            interval_dict['interval_end'].append(interval_end.date())
            interval_dict['modality'].append(modality)
            interval_dict['needs_checking'].append(flagged)

        return pd.DataFrame(interval_dict)

    def _compute_interval(self, row):
        EEG_intervals = self._compute_interval_individual(row, EEG_date_column, 'EEG')
        MRI_intervals = self._compute_interval_individual(row, MRI_date_column, 'MRI')
        intervals = pd.concat((EEG_intervals, MRI_intervals), ignore_index=True)
        intervals = intervals.assign(subject_id=row.subject_id, session_date=row.session_date, diagnosis=row.diagnosis)
        intervals[EEG_date_column] = row[EEG_date_column].date() if pd.notna(row[EEG_date_column]) else row[
            EEG_date_column]
        intervals[MRI_date_column] = row[MRI_date_column].date()

        cols = ['subject_id', 'session_date', 'diagnosis', MRI_date_column, EEG_date_column]
        for col in intervals.columns.to_list():
            if col not in cols:
                cols.append(col)

        intervals = intervals[cols]

        return intervals

    def compute_intervals(self, eligible_participants=None):
        if eligible_participants is None:
            eligible_participants = self.update_eligible_participants_list()

        summary_list = pd.DataFrame()
        for row_idx, row in eligible_participants.iterrows():
            summary_list = summary_list.append(self._compute_interval(row), ignore_index=True)

        return summary_list

    def run(self):
        eligible_participants = self.update_eligible_participants_list()
        intervals = self.compute_intervals(eligible_participants)
        MRI_intervals = intervals.query('modality=="MRI"').reset_index(drop=True).drop(columns=['modality'])
        self._add_delete_worksheet(data=MRI_intervals, ss_title=CBT_ELIGIBLE_PARTICIPANTS_SPREADSHEET,
                                   ws_title='Intervals based on MRI')


# def run():
#     exit(ComputerizedTestsParticipants().run())
#
#     t = A.update_eligible_participants_list()
#     current_eligible_participants = current_eligible_participants[cols]
#
#     participants_to_book = pd.DataFrame()
#     participants_to_check = pd.DataFrame()
#
#     today = datetime.today().date()
#     for idx, row in current_eligible_participants.iterrows():
#         if pd.isna(row.session_date):
#             continue
#         session_date = pd.to_datetime(row.session_date, utc=False).date()
#         for i in range(0, 4):
#             visit_date = session_date + relativedelta.relativedelta(years=i)
#             if i < 3:
#                 report_after = visit_date - relativedelta.relativedelta(months=ACCEPTABLE_GAP_months[0],
#                                                                         days=N_DAYS_BEFORE_TO_REPORT)
#
#                 report_before = visit_date + relativedelta.relativedelta(months=ACCEPTABLE_GAP_months[1])
#
#             else:
#                 report_after = visit_date - relativedelta.relativedelta(months=ACCEPTABLE_GAP_months_end_of_study[0],
#                                                                         days=N_DAYS_BEFORE_TO_REPORT)
#
#                 report_before = visit_date + relativedelta.relativedelta(months=ACCEPTABLE_GAP_months_end_of_study[1])
#
#             previous_visit = None
#             if i > 0:  # After baseline
#                 try:
#                     previous_visit = pd.to_datetime(row[date_col_names_per_year[i - 1]], utc=False).date()
#                     if pd.isna(previous_visit):
#                         previous_visit = None
#
#                 except Exception:
#                     pass
#
#             if previous_visit:
#                 report_after = max(report_after,
#                                    previous_visit + relativedelta.relativedelta(months=min_gap_between_sessions))
#
#             report_grace = report_before + relativedelta.relativedelta(months=Months_of_grace)
#
#             if report_after >= report_before:
#                 new_row = row.copy()
#                 new_row[date_col_names_per_year[i]] = 'Oh noo, please give me some attention...'
#                 participants_to_check = participants_to_check.append(new_row)
#
#             if report_after <= today <= report_grace:
#                 if pd.isna(row[date_col_names_per_year[i]]) or row[date_col_names_per_year[i]].strip == '':
#                     new_row = row[['Fullname', 'subject_id', 'diagnosis', 'session_date']].copy()
#
#                     if i < 3:
#                         collect_after = visit_date - relativedelta.relativedelta(months=ACCEPTABLE_GAP_months[0])
#                         collect_before = visit_date + relativedelta.relativedelta(months=ACCEPTABLE_GAP_months[1])
#
#                     # This would be the end of study.
#                     else:
#                         collect_after = visit_date - relativedelta.relativedelta(
#                             months=ACCEPTABLE_GAP_months_end_of_study[0])
#
#                         collect_before = visit_date + relativedelta.relativedelta(
#                             months=ACCEPTABLE_GAP_months_end_of_study[1])
#
#                     if previous_visit:
#                         collect_after = max(collect_after, previous_visit + relativedelta.relativedelta(
#                             months=min_gap_between_sessions))
#
#                     new_row['Session to collect'] = date_col_names_per_year[i].replace('date', '').strip()
#                     new_row['To collect after'] = collect_after
#                     new_row['To collect before'] = collect_before
#                     participants_to_book = participants_to_book.append(new_row, ignore_index=True)
#
#     if not participants_to_book.empty:
#         participants_to_book = participants_to_book.sort_values(by='To collect before', ignore_index=True,
#                                                                 ascending=True)
#
#     participants_to_book_ws = eligible_ss.worksheet('Participants to book')
#     participants_to_book_ws.clear()
#     set_with_dataframe(worksheet=participants_to_book_ws, dataframe=participants_to_book,
#                        include_index=False, include_column_header=True, allow_formulas=True,
#                        resize=True)
#
#     set_frozen(participants_to_book_ws, rows=1)
#
#     interval = np.array(
#         [x.total_seconds() / 24 / 3600 for x in (participants_to_book['To collect before'] - today).tolist()])
#
#     grace_rows = np.where(interval <= 0)[0]
#     critical_rows = np.where((0 < interval) & (interval <= 28))[0]
#     within_interval_rows = np.where((interval >= 28) & (today > participants_to_book['To collect after']))[0]
#     future_rows = np.where(today < participants_to_book['To collect after'])[0]
#
#     apply_formatting(participants_to_book_ws, rows=grace_rows, fmt=grace_period_format)
#     apply_formatting(participants_to_book_ws, rows=critical_rows, fmt=critical_booking_format)
#     apply_formatting(participants_to_book_ws, rows=within_interval_rows, fmt=within_interval_booking_format)
#     apply_formatting(participants_to_book_ws, rows=future_rows, fmt=future_booking_format)
#
#     if not participants_to_check.empty:
#         try:
#             check_ws = eligible_ss.worksheet('Participants to check')
#             check_ws.clear()
#         except gspread.WorksheetNotFound:
#             check_ws = eligible_ss.add_worksheet('Participants to check', rows=0, cols=0)
#
#         set_with_dataframe(check_ws, dataframe=participants_to_check, resize=True)
#
#     else:
#         try:
#             check_ws = eligible_ss.worksheet('Participants to check')
#             eligible_ss.del_worksheet(check_ws)
#         except Exception:
#             pass


if __name__ == '__main__':
    exit(ComputerizedTestsParticipants().run())
