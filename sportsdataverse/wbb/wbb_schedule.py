import pyarrow.parquet as pq
import pandas as pd
import json
from typing import List, Callable, Iterator, Union, Optional
from sportsdataverse.errors import SeasonNotFoundError
from sportsdataverse.dl_utils import download, underscore

def espn_wbb_schedule(dates=None, groups=50, season_type=None, limit=500) -> pd.DataFrame:
    """espn_wbb_schedule - look up the women's college basketball schedule for a given season

    Args:
        dates (int): Used to define different seasons. 2002 is the earliest available season.
        groups (int): Used to define different divisions. 50 is Division I, 51 is Division II/Division III.
        season_type (int): 2 for regular season, 3 for post-season, 4 for off-season.
        limit (int): number of records to return, default: 500.

    Returns:
        pd.DataFrame: Pandas dataframe containing schedule dates for the requested season.
    """
    if dates is None:
        dates = ''
    else:
        dates = '&dates=' + str(dates)
    if groups is None:
        groups = ''
    else:
        groups = '&groups=' + str(groups)
    if season_type is None:
        season_type = ''
    else:
        season_type = '&seasontype=' + str(season_type)
    url = "http://site.api.espn.com/apis/site/v2/sports/basketball/womens-college-basketball/scoreboard?limit={}{}{}{}".format(limit, dates, groups, season_type)
    resp = download(url=url)

    ev = pd.DataFrame()
    if resp is not None:
        events_txt = json.loads(resp)

        events = events_txt.get('events')
        for event in events:
            event.get('competitions')[0].get('competitors')[0].get('team').pop('links',None)
            event.get('competitions')[0].get('competitors')[1].get('team').pop('links',None)
            if event.get('competitions')[0].get('competitors')[0].get('homeAway')=='home':
                event['competitions'][0]['home'] = event.get('competitions')[0].get('competitors')[0].get('team')
                event['competitions'][0]['home']['score'] = event.get('competitions')[0].get('competitors')[0].get('score')
                event['competitions'][0]['home']['winner'] = event.get('competitions')[0].get('competitors')[0].get('winner')
                event['competitions'][0]['away'] = event.get('competitions')[0].get('competitors')[1].get('team')
                event['competitions'][0]['away']['score'] = event.get('competitions')[0].get('competitors')[1].get('score')
                event['competitions'][0]['away']['winner'] = event.get('competitions')[0].get('competitors')[1].get('winner')
            else:
                event['competitions'][0]['away'] = event.get('competitions')[0].get('competitors')[0].get('team')
                event['competitions'][0]['away']['score'] = event.get('competitions')[0].get('competitors')[0].get('score')
                event['competitions'][0]['away']['winner'] = event.get('competitions')[0].get('competitors')[0].get('winner')
                event['competitions'][0]['home'] = event.get('competitions')[0].get('competitors')[1].get('team')
                event['competitions'][0]['home']['score'] = event.get('competitions')[0].get('competitors')[1].get('score')
                event['competitions'][0]['home']['winner'] = event.get('competitions')[0].get('competitors')[1].get('winner')

            del_keys = ['broadcasts','geoBroadcasts', 'headlines', 'series', 'situation', 'tickets', 'odds']
            for k in del_keys:
                event.get('competitions')[0].pop(k, None)
            if len(event.get('competitions')[0]['notes'])>0:
                event.get('competitions')[0]['notes_type'] = event.get('competitions')[0]['notes'][0].get("type")
                event.get('competitions')[0]['notes_headline'] = event.get('competitions')[0]['notes'][0].get("headline").replace('"','')
            else:
                event.get('competitions')[0]['notes_type'] = ''
                event.get('competitions')[0]['notes_headline'] = ''
            event.get('competitions')[0].pop('notes', None)
            x = pd.json_normalize(event.get('competitions')[0], sep='_')
            x['game_id'] = x['id'].astype(int)
            x['season'] = event.get('season').get('year')
            x['season_type'] = event.get('season').get('type')
            ev = pd.concat([ev,x],axis=0, ignore_index=True)
    ev = pd.DataFrame(ev)
    ev.columns = [underscore(c) for c in ev.columns.tolist()]
    return ev

def espn_wbb_calendar(season=None, ondays=None) -> pd.DataFrame:
    """espn_wbb_calendar - look up the women's college basketball calendar for a given season

    Args:
        season (int): Used to define different seasons. 2002 is the earliest available season.
        ondays (boolean): Used to return dates for calendar ondays

    Returns:
        pd.DataFrame: Pandas dataframe containing
        calendar dates for the requested season.

    Raises:
        ValueError: If `season` is less than 2002.
    """
    if int(season) < 2002:
        raise SeasonNotFoundError("season cannot be less than 2002")
    if ondays is not None:
        url = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/womens-college-basketball/seasons/{}/types/2/calendar/ondays".format(season)
        resp = download(url=url)
        txt = json.loads(resp).get('eventDate').get('dates')
        full_schedule = pd.DataFrame(txt,columns=['dates'])
        full_schedule['dateURL'] = list(map(lambda x: x[:10].replace("-",""),full_schedule['dates']))
        full_schedule['url']="http://site.api.espn.com/apis/site/v2/sports/basketball/womens-college-basketball/scoreboard?dates="
        full_schedule['url']= full_schedule['url'] + full_schedule['dateURL']
    else:
        url = "http://site.api.espn.com/apis/site/v2/sports/basketball/womens-college-basketball/scoreboard?dates={}".format(season)
        resp = download(url=url)
        txt = json.loads(resp)['leagues'][0]['calendar']
        datenum = list(map(lambda x: x[:10].replace("-",""),txt))
        date = list(map(lambda x: x[:10],txt))

        year = list(map(lambda x: x[:4],txt))
        month = list(map(lambda x: x[5:7],txt))
        day = list(map(lambda x: x[8:10],txt))

        data = {"season": season,
                "datetime" : txt,
                "date" : date,
                "year": year,
                "month": month,
                "day": day,
                "dateURL": datenum
        }
        full_schedule = pd.DataFrame(data)
        full_schedule['url']="http://site.api.espn.com/apis/site/v2/sports/basketball/womens-college-basketball/scoreboard?dates="
        full_schedule['url']= full_schedule['url'] + full_schedule['dateURL']
    return full_schedule

