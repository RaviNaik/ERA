<filename>game.go
package main

import (
	"fmt"
	"sync"
)

const (
	AddNewPlayerURL = "https://pzgtmrpgli.execute-api.us-east-1.amazonaws.com/Development/players"
	StartURL        = "https://pzgtmrpgli.execute-api.us-east-1.amazonaws.com/Development/game/start"
	OpenURL         = "https://pzgtmrpgli.execute-api.us-east-1.amazonaws.com/Development/game/open"
	StopURL         = "https://pzgtmrpgli.execute-api.us-east-1.amazonaws.com/Development/game/stop"
	CheckImageURL   = "https://pzgtmrpgli.execute-api.us-east-1.amazonaws.com/Development/check-image"
)

type GameInfo struct {
	IsRunning        bool
	CanJoin          bool
	IsCounterStarted bool
}

type Game struct {
	Mutex  *sync.Mutex
	Helper *Helper
	Info   *GameInfo
	Users  map[string]map[string]interface{}
}

func DefaultGame() *Game {
	return &Game{
		Mutex:  &sync.Mutex{},
		Helper: DefaultHelper(),
		Info: &GameInfo{
			CanJoin: true,
		},
		Users: make(map[string]map[string]interface{}),
	}
}

func (game *Game) Start() (msg string) {
	if game.Info.IsRunning {
		msg = "Game was already started"
	} else {
		game.Info.IsRunning = true
		msg = fmt.Sprintf("You have %d seconds to join. To join, say \"join\"", JoinDelay)
		game.Helper.Request(OpenURL, nil)
	}

	return msg
}

func (game *Game) Join(userID string) (msg string, err error) {

	if userInfo, ok := game.Users[userID]; ok == true {
		// Already Join
		name := userInfo["real_name"].(string)
		msg = fmt.Sprintf("I'm sorry, %s. You've joined the game.", name)
		return msg, nil
	}

	// Haven't join
	userInfo, err := game.Helper.GetUserInfo(userID)
	if err != nil {
		return "", err
	}
	name := userInfo["real_name"].(string)
	game.Mutex.Lock()
	game.Users[userID] = userInfo
	game.Helper.Request(AddNewPlayerURL, map[string]interface{}{
		"username": userInfo["name"].(string),
		"user_id":  userInfo["id"].(string),
	})
	game.Mutex.Unlock()
	return fmt.Sprintf("Congratulations, %s. You're in the game now!", name), nil
}

func (game *Game) Stop() (msg string) {
	result, _ := game.Helper.Request(StopURL, nil)
	msg = result["body"].(string)
	game.Info.IsRunning = false
	return msg
}
