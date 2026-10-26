package command

import (
	"fmt"

	"gopkg.in/alecthomas/kingpin.v2"
)

type RestoreCommand struct {
	*Meta

	GameName   string
	TurnNumber int
}

func (c *RestoreCommand) run(*kingpin.ParseContext) error {
	game, err := c.Meta.RunContext.GameInstallation.AvailableGames.FindGameByName(c.GameName)
	if err != nil {
		return err
	}

	c.Ui.Output(fmt.Sprintf("Restoring turn %v for game %v", c.TurnNumber, game.Name))
	err = game.Restore(c.TurnNumber)
	if err != nil {
		return err
	}

	return nil
}

func (c *RestoreCommand) completion(parseContext *kingpin.ParseContext) error {
	return completionWithGames(c.Meta, parseContext)
}

func ConfigureRestoreCommand(app *kingpin.Application, meta *Meta) (commandName string) {
	commandName = "restore"
	c := &RestoreCommand{Meta: meta}
	cmd := app.Command(commandName, "Restores all backed up files for a given turn for a game.")

	if meta.CompletionOnly {
		cmd.Action(c.completion)
	} else {
		cmd.Action(c.run)
		cmd.Arg("game_name", "Name of the game to restore for").Required().StringVar(&c.GameName)
		cmd.Arg("turn_number", "Restore which turn number?").Required().IntVar(&c.TurnNumber)
	}

	return commandName
}
