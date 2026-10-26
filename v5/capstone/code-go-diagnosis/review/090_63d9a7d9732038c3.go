package core

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"time"
)

// Battle is
type Battle struct {
	TeamID  int
	Time    time.Time
	Teams   []*Team
	Legs    []*Leg
	Current *Leg
}

// NewBattle is
func NewBattle(teamID int, battleTime time.Time) *Battle {
	rand.Seed(time.Now().Unix())
	return &Battle{
		TeamID: teamID,
		Time:   battleTime,
	}
}

// Leg is
type Leg struct {
	Index   int
	Map     *Map
	Teams   []*Team
	TeamMap map[int]*Team
	Players map[int]*Player
	Rounds  []*LegRound
	Current *LegRound
}

// JSON is
func (l *Leg) JSON() string {
	bytes, _ := json.Marshal(l)
	return string(bytes)
}

// LegRound is
type LegRound struct {
	Round  *Round
	Action *Action
}

// StartLeg is
func (b *Battle) StartLeg(legStart *LegStart) *Leg {
	leg := &Leg{
		Index:   len(b.Legs),
		Map:     legStart.Map,
		Teams:   make([]*Team, 0, 2),
		TeamMap: make(map[int]*Team, 2),
	}

	// ensure the current (guest) team is the first item in leg.Teams slice
	otherTeams := make([]*Team, 0)
	for _, team := range legStart.Teams {
		leg.TeamMap[team.ID] = team
		if team.ID == b.TeamID {
			leg.Teams = append(leg.Teams, team)
		} else {
			otherTeams = append(otherTeams, team)
		}
	}
	leg.Teams = append(leg.Teams, otherTeams...)

	// on the coming first leg, update battle's teams' basic information
	// including id, name and players
	if leg.Index == 0 {
		teamCount := len(leg.Teams)
		b.Teams = make([]*Team, teamCount, teamCount)
		for i := 0; i < teamCount; i++ {
			t := leg.Teams[i]
			b.Teams[i] = &Team{
				ID:      t.ID,
				Players: t.Players,
				Force:   t.Force,
			}
		}
	}

	// // init leg players' map
	leg.Players = make(map[int]*Player, 8)
	for _, t := range leg.Teams {
		for _, p := range t.Players {
			leg.Players[p] = &Player{
				TeamID: t.ID,
				ID:     p,
			}
		}
	}

	b.Legs = append(b.Legs, leg)
	b.Current = leg
	return leg
}

// EndLeg is
func (b *Battle) EndLeg(legEnd *LegEnd) error {
	if b.Current == nil {
		return fmt.Errorf("battle state illegal - the battle has no any started legs")
	}
	for _, t1 := range legEnd.Teams {
		t2, ok := b.Current.TeamMap[t1.ID]
		if !ok {
			continue
		}
		t2.Point = t1.Point
	}
	return nil
}

// AddRound is
func (b *Battle) AddRound(round *Round) error {
	if b.Current == nil {
		return fmt.Errorf("battle state illegal - the battle has no any started legs")
	}
	leg := b.Current
	legRound := &LegRound{Round: round}
	leg.Rounds = append(leg.Rounds, legRound)
	leg.Current = legRound

	// update points and remain lives each team
	for _, t1 := range round.Teams {
		t2, ok := leg.TeamMap[t1.ID]
		if !ok {
			continue
		}
		t2.Point = t1.Point
		t2.RemainLife = t1.RemainLife
	}

	// update players' points, sleep and location in current view
	for _, p1 := range round.Players {
		p2, ok := leg.Players[p1.ID]
		if !ok {
			continue
		}
		p2.Point = p1.Point
		p2.Sleep = p1.Sleep
		p2.X = p1.X
		p2.Y = p1.Y
	}
	return nil
}

// CalcAction is
// 0.1 - first version which can work
func (b *Battle) CalcAction() (*Action, error) {
	leg := b.Current
	round := leg.Current.Round
	action := &Action{
		ID: round.ID,
	}
	myPlayers := make([]*Player, 0, DefaultPlayerNum)
	for _, p := range round.Players {
		if p.TeamID == b.TeamID && p.Sleep == 0 {
			myPlayers = append(myPlayers, p)
		}
	}
	l := len(myPlayers)
	action.Actions = make([]*PlayerAction, l, l)
	for i, p := range myPlayers {
		action.Actions[i] = &PlayerAction{
			Team:   p.TeamID,
			Player: p.ID,
			Move:   b.randomMove(),
		}
	}

	return action, nil
}

var allMoves = []string{up, down, left, right, stay}

func (b *Battle) randomMove() []string {
	m := allMoves[rand.Intn(len(allMoves))]
	if m == stay {
		return []string{}
	}
	return []string{m}
}
