import pandas as pd
import pytest
from pivot_stage2_v2.timezone import to_utc,last_sunday

CASES=[]
for y in range(2019,2026):
    CASES.extend([f'{y}-01-15 12:10',f'{y}-07-15 12:10'])
    for month in (3,10):
        day=last_sunday(y,month).strftime('%Y-%m-%d')
        CASES.extend(f'{day} {h}:10' for h in ('01','02','03'))

@pytest.mark.parametrize('timestamp',CASES)
def test_fixed_cet_mapping_across_seasons_and_DST(timestamp):
    before=pd.Timestamp(timestamp)
    actual=to_utc([before],'fixed_cet')[0]
    assert actual==before.tz_localize('UTC')-pd.Timedelta(hours=1)

@pytest.mark.parametrize('year',range(2019,2026))
def test_berlin_ambiguous_and_nonexistent_explicitly_unresolved(year):
    times=[last_sunday(year,m)+pd.Timedelta(hours=2,minutes=10) for m in (3,10)]
    assert to_utc(times,'berlin').isna().all()
    assert not to_utc(times,'fixed_cet').isna().any()

def test_three_hypotheses_summer_difference():
    t=['2024-07-15 12:10']
    assert to_utc(t,'fixed_cet')[0].hour==11
    assert to_utc(t,'berlin')[0].hour==10
    assert to_utc(t,'utc')[0].hour==12
