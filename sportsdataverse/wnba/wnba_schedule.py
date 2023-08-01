from datetime import datetime

import pandas as pd
import polars as pl

from sportsdataverse.dl_utils import download, underscore


def espn_wnba_schedule(dates=None, season_type=None, limit=500, return_as_pandas=False, **kwargs) -> pl.DataFrame:
    """espn_wnba_schedule - look up the WNBA schedule for a given season

    Args:
        dates (int): Used to define different seasons. 2002 is the earliest available season.
        season_type (int): 2 for regular season, 3 for post-season, 4 for off-season.
        limit (int): number of records to return, default: 500.

    Returns:
        pl.DataFrame: Polars dataframe containing schedule dates for the requested season. Returns None if no games
    """
    url = "http://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard"
    params = {"dates": dates, "seasonType": season_type, "limit": limit}
    resp = download(url=url, params=params, **kwargs)

    ev = pd.DataFrame()
    events_txt = resp.json()
    events = events_txt.get("events")
    if events is None:
        return pd.DataFrame() if return_as_pandas else pl.DataFrame()
    if len(events) == 0:
        return pd.DataFrame() if return_as_pandas else pl.DataFrame()

    for event in events:
        event.get("competitions")[0].get("competitors")[0].get("team").pop("links", None)
        event.get("competitions")[0].get("competitors")[1].get("team").pop("links", None)
        if event.get("competitions")[0].get("competitors")[0].get("homeAway") == "home":
            event = __extract_home_away(event, 0, "home")
            event = __extract_home_away(event, 1, "away")
        else:
            event = __extract_home_away(event, 0, "away")
            event = __extract_home_away(event, 1, "home")

        event.get("competitions")[0]["notes_type"] = (
            event.get("competitions")[0]["notes"][0].get("type")
            if len(event.get("competitions")[0]["notes"]) > 0
            else ""
        )
        event.get("competitions")[0]["notes_headline"] = (
            event.get("competitions")[0]["notes"][0].get("headline").replace('"', "")
            if len(event.get("competitions")[0]["notes"]) > 0
            else ""
        )
        event.get("competitions")[0]["broadcast_market"] = (
            event.get("competitions")[0].get("broadcasts", [])[0].get("market", "")
            if len(event.get("competitions")[0].get("broadcasts")) > 0
            else ""
        )
        event.get("competitions")[0]["broadcast_name"] = (
            event.get("competitions")[0].get("broadcasts", [])[0].get("names", [])[0]
            if len(event.get("competitions")[0].get("broadcasts")) > 0
            else ""
        )
        del_keys = ["geoBroadcasts", "headlines", "series", "situation", "tickets", "odds", "leaders"]
        for k in del_keys:
            event.get("competitions")[0].pop(k, None)

        x = pl.from_pandas(pd.json_normalize(event.get("competitions")[0], sep="_"))
        x = x.with_columns(
            game_id=(pl.col("id").cast(pl.Int32)),
            season=(event.get("season").get("year")),
            season_type=(event.get("season").get("type")),
            home_linescores=pl.when(pl.col("status_type_description") == "Postponed")
            .then(None)
            .otherwise(pl.col("home_linescores")),
            away_linescores=pl.when(pl.col("status_type_description") == "Postponed")
            .then(None)
            .otherwise(pl.col("away_linescores")),
        ).with_columns(
            season=pl.col("season").cast(pl.Int32),
            season_type=pl.col("season_type").cast(pl.Int32),
        )
        x = x[[s.name for s in x if s.null_count() != x.height]]
        ev = pd.concat([ev, x.to_pandas()], axis=0, ignore_index=True)
    ev = pl.from_pandas(ev)
    ev.columns = [underscore(c) for c in ev.columns]

    return ev.to_pandas() if return_as_pandas else ev


def __extract_home_away(event, arg1, arg2):
    event["competitions"][0][arg2] = event.get("competitions")[0].get("competitors")[arg1].get("team")
    event["competitions"][0][arg2]["score"] = event.get("competitions")[0].get("competitors")[arg1].get("score")
    event["competitions"][0][arg2]["winner"] = event.get("competitions")[0].get("competitors")[arg1].get("winner")
    # add winner back to main competitors if does not exist
    event["competitions"][0]["competitors"][arg1]["winner"] = (
        event.get("competitions")[0].get("competitors")[arg1].get("winner", False)
    )
    event["competitions"][0][arg2]["linescores"] = (
        event.get("competitions")[0]
        .get("competitors")[arg1]
        .get("linescores", [{"value": 0}, {"value": 0}, {"value": 0}, {"value": 0}])
    )
    # add linescores back to main competitors if does not exist
    event["competitions"][0]["competitors"][arg1]["linescores"] = (
        event.get("competitions")[0]
        .get("competitors")[arg1]
        .get("linescores", [{"value": 0}, {"value": 0}, {"value": 0}, {"value": 0}])
    )
    event["competitions"][0][arg2]["records"] = (
        event.get("competitions")[0]
        .get("competitors")[arg1]
        .get(
            "records",
            [
                {"abbreviation": "Game", "name": "overall", "summary": "0-0", "type": "total"},
                {"abbreviation": "null", "name": "Home", "summary": "0-0", "type": "home"},
                {"abbreviation": "null", "name": "Road", "summary": "0-0", "type": "road"},
            ],
        )
    )
    return event


def espn_wnba_calendar(season=None, ondays=None, return_as_pandas=False, **kwargs) -> pl.DataFrame:
    """espn_wnba_calendar - look up the WNBA calendar for a given season

    Args:
        season (int): Used to define different seasons. 2002 is the earliest available season.
        ondays (boolean): Used to return dates for calendar ondays

    Returns:
        pl.DataFrame: Polars dataframe containing calendar dates for the requested season.

    Raises:
        ValueError: If `season` is less than 2002.
    """
    if ondays is not None:
        full_schedule = __ondays_wnba_calendar(season, **kwargs)
    else:
        url = f"http://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard?dates={season}"
        resp = download(url=url, **kwargs)
        txt = resp.json().get("leagues")[0].get("calendar")
        datenum = list(map(lambda x: x[:10].replace("-", ""), txt))
        date = list(map(lambda x: x[:10], txt))
        year = list(map(lambda x: x[:4], txt))
        month = list(map(lambda x: x[5:7], txt))
        day = list(map(lambda x: x[8:10], txt))
        data = {
            "season": season,
            "datetime": txt,
            "date": date,
            "year": year,
            "month": month,
            "day": day,
            "dateURL": datenum,
        }
        full_schedule = pl.DataFrame(data)
        full_schedule = full_schedule.with_columns(
            url="http://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard?dates=" + pl.col("dateURL")
        )
    return full_schedule.to_pandas() if return_as_pandas else full_schedule


def __ondays_wnba_calendar(season, **kwargs):
    url = (
        f"https://sports.core.api.espn.com/v2/sports/basketball/leagues/wnba/seasons/{season}/types/2/calendar/ondays"
    )
    resp = download(url=url, **kwargs)
    txt = resp.json().get("eventDate").get("dates")
    result = pl.DataFrame(txt, schema=["dates"])
    result = result.with_columns(dateURL=pl.col("dates").str.slice(0, 10))
    result = result.with_columns(
        url="http://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard?dates=" + pl.col("dateURL")
    )

    return result


def most_recent_wnba_season():
    today = datetime.date(datetime.now())
    return today.year if today.month >= 5 else today.year - 1
