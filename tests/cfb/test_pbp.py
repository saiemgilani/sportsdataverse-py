from sportsdataverse.cfb.cfb_pbp import CFBPlayProcess
import pandas as pd
import pytest
import logging
from sportsdataverse.cfb.model_vars import *

LOGGER = logging.getLogger(__name__)
logging.basicConfig()

@pytest.fixture()
def generated_data():
    test = CFBPlayProcess(gameId = 401301025)
    test.espn_cfb_pbp()
    test.run_processing_pipeline()
    yield test

@pytest.fixture()
def box_score(generated_data):
    box = generated_data.create_box_score()
    yield box

def test_basic_pbp(generated_data):
    assert generated_data.json != None

    generated_data.run_processing_pipeline()
    assert len(generated_data.plays_json) > 0
    assert generated_data.ran_pipeline == True
    assert isinstance(generated_data.plays_json, pd.DataFrame)

def test_adv_box_score(box_score):
    assert box_score != None
    assert len(set(box_score.keys()).difference({"win_pct","pass","team","situational","receiver","rush","receiver","defensive","turnover","drives"})) == 0

def test_havoc_rate(generated_data):
    generated_data.run_processing_pipeline()
    box_score = generated_data.create_box_score()


    defense_home = box_score["defensive"][0]
    # print(defense_home)
    pd = defense_home.get("pass_breakups", 0)
    home_int = defense_home.get("def_int", 0)
    tfl = defense_home.get("TFL", 0)
    fum = defense_home.get("fumbles", 0)
    plays = defense_home.get("scrimmage_plays", 0)

    # mask = (generated_data.plays_json.statYardage < 0) & (generated_data.plays_json.penalty_flag == False) & (generated_data.plays_json["start.team.id"] != 2567)
    # LOGGER.info(generated_data.plays_json[mask][["id", "text", "statYardage", "havoc", "start.down", "start.yardsToEndzone", "end.down", "end.yardsToEndzone", "int", "forced_fumble"]].to_json(orient = "records", indent = 2))
    LOGGER.info(generated_data.plays_json[(generated_data.plays_json.havoc == True) & (generated_data.plays_json.penalty_flag == False) & (generated_data.plays_json["start.team.id"] != 2567)][["id", "text", "statYardage", "havoc", "start.down", "start.yardsToEndzone", "end.down", "end.yardsToEndzone", "int", "forced_fumble", "TFL", "TFL_pass", "TFL_rush"]].to_json(orient = "records", indent = 2))
    LOGGER.info({
        "pd": pd,
        "home_int": home_int,
        "tfl": tfl,
        "fum": fum
    })

    assert plays > 0
    assert defense_home["havoc_total"] == (pd + home_int + tfl + fum)
    assert round(defense_home["havoc_total_rate"], 4) == round(((pd + home_int + tfl + fum) / plays), 4)

@pytest.fixture()
def dupe_fsu_play_base():
    test = CFBPlayProcess(gameId = 401411109)
    test.espn_cfb_pbp()
    test.run_processing_pipeline()
    yield test.plays_json

def test_fsu_play_dedupe(dupe_fsu_play_base):
    target_strings = [
        {
            "text": "Jordan Travis pass intercepted Rance Conner return for no gain to the FlaSt 45",
            "down": 3,
            "distance": 9,
            "yardsToEndzone": 74
        },
        {
            "down" : 4,
            "text": "Malik Cunningham pass incomplete to Tyler Hudson",
            "distance": 2,
            "yardsToEndzone": 45
        }
    ]

    regression_cases = [
        {
            "text" : "Alex Mastromanno punt for 52 yds , Braden Smith returns for no gain to the Lvile 37",
            "down" : 4,
            "distance" : 9,
            "yardsToEndzone" : 89
        }
    ]

    for item in target_strings:
        print(f"Checking known test cases for dupes for play_text '{item}'")
        assert len(dupe_fsu_play_base[
            (dupe_fsu_play_base["text"] == item["text"])
            & (dupe_fsu_play_base["start.down"] == item["down"])
            & (dupe_fsu_play_base["start.distance"] == item["distance"])
            & (dupe_fsu_play_base["start.yardsToEndzone"] == item["yardsToEndzone"])
        ]) == 1
        print(f"No dupes for play_text '{item}'")


    for item in regression_cases:
        print(f"Checking non-dupe base cases for dupes for play_text '{item}'")
        assert len(dupe_fsu_play_base[
            (dupe_fsu_play_base["text"] == item["text"])
            & (dupe_fsu_play_base["start.down"] == item["down"])
            & (dupe_fsu_play_base["start.distance"] == item["distance"])
            & (dupe_fsu_play_base["start.yardsToEndzone"] == item["yardsToEndzone"])
        ]) == 1
        print(f"confirmed no dupes for regression case of play_text '{item}'")

@pytest.fixture()
def iu_play_base():
    test = CFBPlayProcess(gameId = 401426563)
    test.espn_cfb_pbp()
    test.run_processing_pipeline()
    yield test

@pytest.fixture()
def dupe_iu_play_base(iu_play_base):
    yield iu_play_base.plays_json

def test_iu_play_dedupe(dupe_iu_play_base):
    target_strings = [
        {
            "text": "A. Reed pass,to J. Beljan for 26 yds for a TD, (B. Narveson KICK)",
            "down": 2,
            "distance": 9,
            "yardsToEndzone": 26
        }
    ]

    elimination_strings = [
        {
            "text" : "Austin Reed pass complete to Joey Beljan for 26 yds for a TD",
            "down": 2,
            "distance": 9,
            "yardsToEndzone": 26
        }
    ]

    for item in target_strings:
        print(f"Checking known test cases for dupes for play_text '{item}'")
        assert len(dupe_iu_play_base[
            (dupe_iu_play_base["text"] == item["text"])
            & (dupe_iu_play_base["start.down"] == item["down"])
            & (dupe_iu_play_base["start.distance"] == item["distance"])
            & (dupe_iu_play_base["start.yardsToEndzone"] == item["yardsToEndzone"])
        ]) == 1
        print(f"No dupes for play_text '{item}'")

    for item in elimination_strings:
        print(f"Checking for strings that should have been removed by dupe check for play_text '{item}'")
        assert len(dupe_iu_play_base[
            (dupe_iu_play_base["text"] == item["text"])
            & (dupe_iu_play_base["start.down"] == item["down"])
            & (dupe_iu_play_base["start.distance"] == item["distance"])
            & (dupe_iu_play_base["start.yardsToEndzone"] == item["yardsToEndzone"])
        ]) == 0
        print(f"Confirmed no values for play_text '{item}'")

@pytest.fixture()
def iu_play_base_box(iu_play_base):
    box = iu_play_base.create_box_score()
    yield box

def test_expected_turnovers(iu_play_base_box):
    defense_home = iu_play_base_box["defensive"][1]
    def_home_team = defense_home.get('def_pos_team', 'NA')
    away_pd = iu_play_base_box['turnover'][0].get("pass_breakups", 0)
    away_off_int = iu_play_base_box['turnover'][0].get("Int", 0)
    away_fum = iu_play_base_box['turnover'][0].get("total_fumbles", 0)

    away_exp_xTO = (0.22 * (away_pd + away_off_int)) + (0.5 * away_fum)
    away_actual_xTO = iu_play_base_box['turnover'][0].get('expected_turnovers')
    away_team = iu_play_base_box['turnover'][0].get('pos_team', "NA")

    defense_away = iu_play_base_box["defensive"][0]
    def_away_team = defense_away.get('def_pos_team', 'NA')
    home_pd = iu_play_base_box['turnover'][1].get("pass_breakups", 0)
    home_off_int = iu_play_base_box['turnover'][1].get("Int", 0)
    home_fum = iu_play_base_box['turnover'][1].get("total_fumbles", 0)

    home_exp_xTO = (0.22 * (home_pd + home_off_int)) + (0.5 * home_fum)
    home_actual_xTO = iu_play_base_box['turnover'][1].get('expected_turnovers')
    home_team = iu_play_base_box['turnover'][1].get('pos_team', "NA")

    print(f"home team: {home_team} vs def {def_away_team} - fum: {home_fum}, int: {home_off_int}, pd: {home_pd} -> xTO: {home_exp_xTO}")
    print(f"away off {away_team} vs def {def_home_team} - fum: {away_fum}, int: {away_off_int}, pd: {away_pd} -> xTO: {away_exp_xTO}")
    assert round(away_exp_xTO, 4) == round(away_actual_xTO, 4)
    assert round(home_exp_xTO, 4) == round(home_actual_xTO, 4)


def test_onside_kickoff_recovery():
    test_fsu_23 = CFBPlayProcess(gameId = 401525493)
    test_fsu_23.espn_cfb_pbp()
    test_fsu_23.run_processing_pipeline()

    target_plays_fsu_23 = test_fsu_23.plays_json[
        (test_fsu_23.plays_json["text"] == "Ryan Fitzgerald on-side kick recovered by Florida State at the FSU 49")
    ]

    # winning team kicks onside
    LOGGER.info("---- ONSIDE KICK (FSU/SOMISS 2023) ----")
    LOGGER.info(target_plays_fsu_23.iloc[0]["type.text"])
    LOGGER.info(target_plays_fsu_23.iloc[0]["kickoff_onside"])
    LOGGER.info(target_plays_fsu_23.iloc[0]["change_of_pos_team"])
    LOGGER.info(target_plays_fsu_23.iloc[0]["change_of_poss"])
    LOGGER.info(target_plays_fsu_23.iloc[0]["wp_after"])
    LOGGER.info(target_plays_fsu_23.iloc[0]["wpa"])
    LOGGER.info(target_plays_fsu_23.iloc[0]["pos_score_diff_end"])
    assert float(target_plays_fsu_23.iloc[0]["wp_after"]) < 0.1
    assert float(target_plays_fsu_23.iloc[0]["wpa"]) < 0.9

    test_gatech_15 = CFBPlayProcess(gameId = 400756922)
    test_gatech_15.espn_cfb_pbp()
    test_gatech_15.run_processing_pipeline()

    target_plays_gatech_15 = test_gatech_15.plays_json[
        (test_gatech_15.plays_json["text"] == "Harrison Butker on-side kick recovered by GEORGIA TECH at the NDame 43")
    ]

    # losing team kicks onside
    LOGGER.info("---- ONSIDE KICK (GT/ND 2015) ----")
    LOGGER.info(target_plays_gatech_15.iloc[0]["type.text"])
    LOGGER.info(target_plays_gatech_15.iloc[0]["kickoff_onside"])
    LOGGER.info(target_plays_gatech_15.iloc[0]["change_of_pos_team"])
    LOGGER.info(target_plays_gatech_15.iloc[0]["change_of_poss"])
    LOGGER.info(target_plays_gatech_15.iloc[0]["wp_after"])
    LOGGER.info(target_plays_gatech_15.iloc[0]["wpa"])
    LOGGER.info(target_plays_gatech_15.iloc[0]["pos_score_diff_end"])
    assert float(target_plays_gatech_15.iloc[0]["wp_after"]) > 0.9
    assert float(target_plays_gatech_15.iloc[0]["wpa"]) < 0.1

def test_play_order():
    test = CFBPlayProcess(gameId = 401525825)
    test.espn_cfb_pbp()
    test.run_processing_pipeline()
    
    should_be_first = test.plays_json[
        (test.plays_json["text"] == "Tahj Brooks run for 1 yd to the WYO 6")
        & (test.plays_json["start.down"] == 2)
        & (test.plays_json["start.distance"] == 3)
        & (test.plays_json["start.yardsToEndzone"] == 7)
    ]

    should_be_next = test.plays_json[
        (test.plays_json["text"] == "Tahj Brooks 6 Yd Run (Gino Garcia Kick)")
    ]

    pbp_ot = test.plays_json[
        (test.plays_json["period.number"] == 5)
    ]
    LOGGER.info(pbp_ot[["id", "sequenceNumber", "period", "start.down", "start.distance", "text"]])

    assert int(should_be_first.iloc[0]["sequenceNumber"]) + 1 == int(should_be_next.iloc[0]["sequenceNumber"])
    assert int(should_be_first.iloc[0]["game_play_number"]) + 1 == int(should_be_next.iloc[0]["game_play_number"])

def test_explosive_play_count():
    test = CFBPlayProcess(gameId = 401525500)
    test.espn_cfb_pbp()
    test.run_processing_pipeline()

    box = test.create_box_score()
    
    fsu_expl_total = box['team'][0]['EPA_explosive']
    LOGGER.info(fsu_expl_total)

    fsu_expl_plays = test.plays_json[
        (test.plays_json["pos_team"] == 52)
        & ((test.plays_json["EPA"] >= 1.8))
    ]
    LOGGER.info(fsu_expl_plays[["id", "text", "statYardage", "pass", "rush", "EPA", "EPA_explosive"]])

    fsu_naive_expl_plays = test.plays_json[
        (test.plays_json["pos_team"] == 52)
        & (test.plays_json["statYardage"] >= 15)
        # & (test.plays_json["scrimmage_play"] == True)
    ]
    LOGGER.info(fsu_naive_expl_plays[["id", "text", "statYardage", "pass", "rush", "EPA", "EPA_explosive"]])
    LOGGER.info(len(fsu_naive_expl_plays))

    bc_naive_expl_plays = test.plays_json[
        (test.plays_json["pos_team"] != 52)
        & (test.plays_json["statYardage"] >= 15)
        # & (test.plays_json["scrimmage_play"] == True)
    ]
    LOGGER.info(bc_naive_expl_plays[["id", "text", "statYardage", "pass", "rush", "EPA", "EPA_explosive"]])
    LOGGER.info(len(bc_naive_expl_plays))

    # assert fsu_expl_total == len(fsu_expl_plays)

def test_spread_available():
    test = CFBPlayProcess(gameId = 401525519)
    test.espn_cfb_pbp()
    json_dict_stuff = test.run_processing_pipeline()

    # assert that pickcenter is dead for all games
    assert len(json_dict_stuff["pickcenter"]) == 0
    assert test.plays_json.loc[0, "gameSpreadAvailable"] == True

def test_def_fumbles_lost():
    test = CFBPlayProcess(gameId = 401525530)
    test.espn_cfb_pbp()
    json_dict_stuff = test.run_processing_pipeline()

    box_score = test.create_box_score()
    LOGGER.info(box_score['turnover'][0])

    fsu_fumbles_lost = box_score['turnover'][0]['fumbles_lost']
    fsu_fumbles_recovered = box_score['turnover'][0]['fumbles_recovered']
    fsu_fumbles_total = box_score['turnover'][0]['total_fumbles']

    fsu_fum_plays = test.plays_json[
        (test.plays_json["pos_team"] == 52)
        & (test.plays_json["fumble_lost"] == True)
    ]
    LOGGER.info(fsu_fum_plays[["pos_team", "text"]]) #, "fumble_lost", "fumble_vec", "fumble_recovered"]])

    assert fsu_fumbles_total == 1
    assert fsu_fumbles_lost == 0
    assert fsu_fumbles_recovered == 1

def test_ou_tul_bad_spread():
    test = CFBPlayProcess(gameId = 401287894)
    test.espn_cfb_pbp()
    json_dict_stuff = test.run_processing_pipeline()

    # LOGGER.info(json_dict_stuff["pickcenter"])

    # assert len(json_dict_stuff["pickcenter"]) == 0
    assert test.plays_json.loc[0, "gameSpreadAvailable"] == True
    assert test.plays_json.loc[0, "homeTeamSpread"] >= 31.0
    assert test.plays_json.loc[0, "homeTeamId"] == 201


def test_bad_wp_after_situations():
    test = CFBPlayProcess(gameId = 401551786) # Ohio St/Mich: 401520434 vs BC/SMU: 401551750
    test.espn_cfb_pbp()
    json_dict_stuff = test.run_processing_pipeline()

    plays = test.plays_json

    plays["lead_play_text"] = plays["text"].shift(-1)

    bad_wpa_play = plays[
        plays["text"].isin([
            "Michigan Penalty, Unsportsmanlike Conduct (Jaylen Harrell) to the MICH 11 for a 1ST down",
            "[NHSG] Kneel down by MCCARTHY, J.J. at MIC9 (team loss of 2), clock 00:00.",
            "Jesse Mirco punt for 43 yds, downed at the MICH 36",
            "Tommy Doman punt for 49 yds, downed at the OSU 2",
            "Ryan Bujcevski punt blocked by TEAM blocked by TEAM Bujcevski, Ryan punt 29 yards to the SMU44, recovered by BOSTONCOLL # at SMU44 (blocked by TEAM).",
            "Kevin Jennings pass incomplete, broken up by #",
            "Jalen Milroe run for 1 yd to the MICH 2"
        ])
    ]

    bad_wpa_play["proper_time_set"] = bad_wpa_play["start.adj_TimeSecsRem"] >= bad_wpa_play["end.adj_TimeSecsRem"]

    search_cols = sorted(list(set(wp_start_columns + wp_end_columns + ["end.ExpScoreDiff", "start.ExpScoreDiff"])))
    LOGGER.info(bad_wpa_play[["id", "text", "lead_play_text", "change_of_poss", "change_of_pos_team", "wp_after_case", "wp_before", "wp_after", "proper_time_set", "game_play_number"] + search_cols].to_json(orient = "records", indent = 2))

    assert bad_wpa_play.proper_time_set.all()

def test_available_yards():
    test = CFBPlayProcess(gameId = 401635525) # Ohio St/Mich: 401520434 vs BC/SMU: 401551750
    test.espn_cfb_pbp()
    json_dict_stuff = test.run_processing_pipeline()
    box = test.create_box_score()

    # plays = test.plays_json

    LOGGER.info(box)


def test_bugged_pass_yards():
    test = CFBPlayProcess(gameId = 401628456)     # known bugged game - 2024 W1: Idaho vs Oregon
    test.espn_cfb_pbp()
    json_dict_stuff = test.run_processing_pipeline()
    
    plays = test.plays_json
    bad_yards_play = plays[
        ((plays['text'].str.contains(" pass complete ")) & (plays['start.team.id'] == 70)) # Idaho passing yards
        |  ((plays['text'].str.contains(" sacked ")) & (plays['start.team.id'] == 70))
    ]
    LOGGER.info(bad_yards_play[["id", "text", "yds_receiving", "statYardage", "start.yardsToEndzone", "end.yardsToEndzone", "yds_sacked"]].to_json(orient = "records", indent = 2))
    
    box = test.create_box_score()
    LOGGER.info(box['pass'][0])
    LOGGER.info(box['rush'][0])

    assert box['pass'][0]['Yds'] == (168 - 25) # make sure a bugged game matches the right total
    assert box['rush'][0]['Yds'] == 47 # rush totals should not have been changed

    # make sure sack yardage is accounted for
    assert list(filter(lambda x: x['passer_player_name'] == "Dillon Gabriel", box['pass']))[0]['Yds'] == (380 - 23) # make sure a bugged game matches the right total


    # known good game - 2024 W1: GAST vs Georgia Tech
    good = CFBPlayProcess(gameId = 401634302)     # known bugged game - 2024 W1: Idaho vs Oregon
    good.espn_cfb_pbp()
    good_json = good.run_processing_pipeline()
    
    good_plays = good.plays_json
    good_yards_play = good_plays[
        (good_plays['text'].str.contains(" pass complete ")) & (good_plays['start.team.id'] == 59) # GT passing yards
    ]
    LOGGER.info(good_yards_play[["id", "text", "yds_receiving", "statYardage", "start.yardsToEndzone", "end.yardsToEndzone"]].to_json(orient = "records", indent = 2))

    good_box = good.create_box_score()
    LOGGER.info(good_box['pass'][1])
    LOGGER.info(good_box['rush'][1])

    assert good_box['pass'][1]['Yds'] == -1 # make sure a non-bugged game matches the right total
    assert good_box['rush'][1]['Yds'] == 20 # rush totals should not have been changed

    # edge case: completed pass, fumble, recovery
    edge = CFBPlayProcess(gameId = 401634169)

    edge.espn_cfb_pbp()
    edge_json = edge.run_processing_pipeline()
    
    edge_plays = edge.plays_json
    edge_yards_play = edge_plays[
        (edge_plays['text'].str.contains("Hudson Card pass complete to Drew Biber for 2 yds fumbled, forced by Maddix Blackwell, recovered by INST Garret Ollendieck G. Ollendieck return for 0 yds"))
    ]
    LOGGER.info(edge_yards_play[["id", "text", "yds_receiving", "statYardage", "start.yardsToEndzone", "end.yardsToEndzone"]].to_json(orient = "records", indent = 2))
    assert edge_yards_play.loc[edge_yards_play.index[0], 'yds_receiving'] == 2


def test_neb_24wk1():
    test = CFBPlayProcess(gameId = 401628454)
    test.espn_cfb_pbp()
    json_dict_stuff = test.run_processing_pipeline()

    box = test.create_box_score()
    LOGGER.info(box['pass'][0])
    assert box['pass'][0]['Yds'] == (238.0 - 6.0)
    # LOGGER.info(box['rush'][0])
    
    # plays = test.plays_json
    # bad_yards_play = plays[
    #     (plays['text'].str.contains(" pass complete ")) & (plays['text'].str.contains('Raiola '))
    # ]
    # LOGGER.info(bad_yards_play[["id", "text", "yds_passing", "yds_receiving", "statYardage", "start.yardsToEndzone", "end.yardsToEndzone", "yds_sacked"]].to_json(orient = "records", indent = 2))
    