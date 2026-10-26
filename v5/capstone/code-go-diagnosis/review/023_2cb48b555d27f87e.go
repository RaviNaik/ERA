package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/gob"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"math/rand"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/testgrid/hackathon/pkg/hackupdater"
	hackimage "github.com/GoogleCloudPlatform/testgrid/hackathon/pkg/image"
	"github.com/GoogleCloudPlatform/testgrid/util/gcs"
	"github.com/sirupsen/logrus"
)

// options configures the updater
type options struct {
	config  gcs.Path // gs://path/to/config/proto
	confirm bool
	group   string
	url     string

	debug bool
	trace bool
}

const defaultURL = "https://hackathon-dot-k8s-testgrid.appspot.com/r/k8s-testgrid-hackathon/everyone#michelle192837&width=40&sort-by-name="

// gatherOptions reads options from flags
func gatherFlagOptions(fs *flag.FlagSet, args ...string) options {
	var o options
	fs.Var(&o.config, "config", "gs://path/to/config.pb")
	fs.BoolVar(&o.confirm, "confirm", false, "Upload data if set")
	fs.StringVar(&o.group, "test-group", "michelle192837", "Only update named group if set")
	fs.StringVar(&o.url, "url", defaultURL, "Navigate to specified URL after advancing game")

	fs.BoolVar(&o.debug, "debug", true, "Log debug lines if set")
	fs.BoolVar(&o.trace, "trace", false, "Log trace and debug lines if set")

	fs.Parse(args)
	return o
}

// gatherOptions reads options from flags
func gatherOptions() options {
	return gatherFlagOptions(flag.CommandLine, os.Args[1:]...)
}

type setting struct {
	w     int
	h     int
	mines int
}

var settings = []setting{
	{9, 9, 10},   // Beginner
	{16, 16, 40}, // Intermediate
	{30, 16, 99}, // Expert
}

type Cell struct {
	Mine  bool
	Open  bool
	Flag  bool
	Maybe bool
}

func (c Cell) String() string {
	switch {
	case c.Flag:
		return "!"
	case c.Open && c.Mine:
		return "*"
	case !c.Open:
		return "?"
	default:
		return " "
	}
}

type board [][]*Cell

func (b board) over() *image.Point {
	for x, cells := range b {
		for y, cell := range cells {
			if cell.Mine && cell.Open {
				return &image.Point{x, y}
			}
		}
	}
	return nil
}

func (b board) String() string {
	var sb strings.Builder
	sb.WriteString("\n")
	sb.WriteString("+")
	for x := 0; x < b.width(); x++ {
		sb.WriteString("-")
	}
	sb.WriteString("+")
	sb.WriteString("\n")
	for y := 0; y < b.height(); y++ {
		sb.WriteString("|")
		for x := 0; x < b.width(); x++ {
			ch := b[x][y].String()
			if ch == " " {
				if n := mines(b.neighbors(x, y)); n > 0 {
					ch = strconv.Itoa(n)
				}
			}
			sb.WriteString(ch)
		}
		sb.WriteString("|")
		sb.WriteString("\n")
	}
	sb.WriteString("+")
	for x := 0; x < b.width(); x++ {
		sb.WriteString("-")
	}
	sb.WriteString("+")
	sb.WriteString("\n")
	return sb.String()
}

func generate(s setting) board {
	cells := make([][]*Cell, 0, s.w)
	for col := 0; col < s.w; col++ {
		cells = append(cells, make([]*Cell, s.h))
		for row := 0; row < s.h; row++ {
			cells[col][row] = &Cell{}
		}
	}
	// TODO(fejta): maybe better distro, aka ensure solvable.
	for s.mines > 0 {
		rw, rh := rand.Intn(s.w), rand.Intn(s.h)
		cell := cells[rw][rh]
		if cell.Mine {
			continue
		}
		cell.Mine = true
		s.mines--
	}

	return cells
}

func (b board) width() int {
	return len(b)
}

func (b board) height() int {
	if b.width() == 0 {
		return 0
	}
	return len(b[0])
}

func (b board) open(x, y int) bool {
	cell := b.cell(x, y)
	if cell == nil {
		return false
	}
	if cell.Open {
		return false
	}
	b[x][y].Open = true
	neighbors := b.neighbors(x, y)
	if mines(neighbors) > 0 {
		return true
	}
	for dx := -1; dx <= 1; dx++ {
		for dy := -1; dy <= 1; dy++ {
			b.open(x+dx, y+dy)
		}
	}
	return true
}

func (b board) cell(x, y int) *Cell {
	if x < 0 || x >= b.width() {
		return nil
	}
	if y < 0 || y >= b.height() {
		return nil
	}
	return b[x][y]
}

func (b board) neighbors(x, y int) []*Cell {
	cells := make([]*Cell, 0, 8)
	for dx := -1; dx <= 1; dx++ {
		for dy := -1; dy <= 1; dy++ {
			if dx == dy && dy == 0 {
				continue
			}
			cell := b.cell(dx+x, dy+y)
			if cell == nil {
				continue
			}
			cells = append(cells, cell)
		}
	}
	return cells
}

func mines(neighbors []*Cell) int {
	var found int
	for _, cell := range neighbors {
		if cell.Mine {
			found++
		}
	}
	return found
}

var (
	gray       = color.RGBA{192, 192, 192, 0xff}
	blue       = color.RGBA{0, 0, 0xff, 0xff}
	green      = color.RGBA{0, 128, 0, 0xff}
	red        = color.RGBA{0xff, 0, 0, 0xff}
	black      = color.RGBA{0, 0, 0, 0xff}
	lightGreen = color.RGBA{0xaa, 0xee, 0xbb, 0xff}

	originalColors = []color.RGBA{
		gray,                            // 0
		blue,                            // 1
		green,                           // 2
		red,                             // 3
		color.RGBA{0, 0, 128, 0xff},     // 4
		color.RGBA{128, 0, 0, 0xff},     // 5
		color.RGBA{0, 128, 128, 0xff},   // 6
		color.RGBA{0, 0, 0, 0xff},       // 7
		color.RGBA{128, 128, 128, 0xff}, // 8
	}
)

var (
	encoding = base64.URLEncoding
)

func encode(what interface{}) string {
	var sb strings.Builder
	enc := gob.NewEncoder(&sb)
	if err := enc.Encode(what); err != nil {
		panic(err)
	}
	return encoding.EncodeToString([]byte(sb.String()))
}

const (
	removeFlag = "Unflag cell"
	addFlag    = "Flag cell as mined"
	unmined    = "Clean and swept"
	danger     = "Cell is mined, watch out!"
	unknown    = "Sweep for mines"
	win        = "You Win!"
	lose       = "Too bad, try again!"
)

func (b board) render(act *action) *hackimage.Image {
	w, h := b.width(), b.height()
	rect := image.Rect(0, 0, w, h)

	flagging := act != nil && act.Flag

	img := hackimage.New(rect)
	over := b.over()
	var msg string
	for x := 0; x < w; x++ {
		for y := 0; y < h; y++ {
			cell := b.cell(x, y)
			var c color.Color
			a := action{
				Board:      &b,
				Open:       &image.Point{x, y},
				Flag:       flagging,
				Start:      act.Start,
				Difficulty: act.Difficulty,
				Emoji:      act.Emoji,
			}
			id := encode(&a)
			switch {
			case cell.Flag:
				if flagging {
					msg = removeFlag
				} else {
					msg = danger
				}
				var icon string
				if cell.Maybe {
					icon = act.icon("❓")
				} else if over == nil || cell.Mine {
					icon = act.icon("🧨")
				} else {
					icon = act.icon("❌")
					msg = "No mine here"
				}
				c = hackimage.MetaColor(gray, icon, msg, id)
			case !cell.Open && (over == nil || !cell.Mine):
				if flagging {
					msg = addFlag
				} else {
					msg = unknown
				}
				c = hackimage.MetaColor(gray, ".", msg, id)
			case cell.Mine:
				var icon string
				if over == nil || over.X != x || over.Y != y {
					c = black
					icon = act.icon("💣")
				} else {
					c = red
					icon = act.icon("💥")
				}
				c = hackimage.MetaColor(c, icon, lose, id)
			default:
				icon := " "
				if m := mines(b.neighbors(x, y)); m > 0 {
					icon = strconv.Itoa(m)
					c = green
				} else {
					c = lightGreen
				}
				c = hackimage.MetaColor(c, icon, unmined, id)
			}
			img.Set(x, y, c)
		}
	}
	return img
}

func toolbar(act *action) *hackimage.Image {
	var noop string
	var w int
	if act != nil && act.Board != nil {
		w = act.Board.width()
		noop = encode(act)
	}
	if w < minWidth {
		w = minWidth
	}
	img := hackimage.New(image.Rect(0, 0, w, 2))
	id := encode(&action{
		New:        true,
		Difficulty: act.Difficulty,
		Emoji:      act.Emoji,
	})
	mid := w / 2
	var seconds int64
	if act != nil && act.Board != nil {
		var remain int
		var closed int
		var boom bool
		for _, cells := range *act.Board {
			for _, cell := range cells {
				if cell.Flag {
					remain--
				}
				if cell.Mine {
					remain++
				}
				if !cell.Open && !cell.Mine {
					closed++
				}
				if cell.Open && cell.Mine {
					boom = true
				}
			}
		}

		var icon string
		switch {
		case boom:
			icon = act.icon("😖")
		case closed == 0:
			icon = act.icon("😎")
		default:
			icon = act.icon("😐")
		}
		img.Set(mid, 0, hackimage.MetaColor(gray, icon, "New game", id))

		img.Set(0, 0, hackimage.MetaColor(black, fmt.Sprintf("%03d", remain), fmt.Sprintf("%03d mines remain", remain), noop))
		var msg string
		if act.Flag {
			msg = "Stop flagging"
		} else {
			msg = "Flag cells with mines"
		}
		id := encode(&action{
			Board:      act.Board,
			Flag:       !act.Flag,
			Start:      act.Start,
			Difficulty: act.Difficulty,
			Emoji:      act.Emoji,
		})
		var c color.Color
		if act.Flag {
			c = blue
		} else {
			c = gray
		}
		img.Set(1, 0, hackimage.MetaColor(c, act.icon("⛳"), msg, id))
		id = encode(&action{
			Board:      act.Board,
			Flag:       act.Flag,
			Start:      act.Start,
			Difficulty: act.Difficulty,
			Emoji:      !act.Emoji,
		})
		if act.Emoji {
			icon = "A"
			msg = "Ascii graphics"
		} else {
			icon = "🤪"
			msg = "Fancy graphics"
		}
		img.Set(2, 0, hackimage.MetaColor(c, icon, msg, id))
		seconds = time.Now().Unix() - act.Start
		if seconds > 999 {
			logrus.WithField("seconds", seconds).Warning("Truncating time")
			seconds = 999
		}
	} else {
		img.Set(mid, 0, hackimage.MetaColor(gray, act.icon("😐"), "New game", id))
		seconds = 0
	}
	if w > 9 {
		drawSettings(act, img, mid+2)
	} else {
		drawSettings(act, img, mid+1)
	}

	for x := 0; x < w; x++ {
		img.Set(x, 1, hackimage.MetaColor(black, "=", "", encode(act)))
	}

	s := fmt.Sprintf("%03d", seconds)
	img.Set(w-1, 0, hackimage.MetaColor(black, s, s+" seconds remaining", noop))
	return img
}

func drawSettings(act *action, img draw.Image, x int) {
	for i, diff := range []string{"Beginner", "Intermediate", "Expert"} {
		set := settings[i]
		var a *action
		var c color.Color
		if i == act.Difficulty {
			c = blue
			a = act
		} else {
			c = gray
			a = &action{
				New:        true,
				Difficulty: i,
				Emoji:      act.Emoji,
			}
		}
		id := encode(a)
		var icon string
		switch i {
		case 0:
			icon = act.icon("👶")
		case 1:
			icon = act.icon("👧")
		default:
			icon = act.icon("👵")
		}
		c = hackimage.MetaColor(c, icon, fmt.Sprintf("Change difficulty to %s: %dx%d (%d mines)", diff, set.w, set.h, set.mines), id)
		img.Set(x+i, 0, c)
	}
}

type action struct {
	Board      *board
	Flag       bool
	New        bool
	Open       *image.Point
	Start      int64
	Difficulty int
	Emoji      bool
}

var emojis = map[string]string{
	"😖": "N",
	"😎": "N",
	"😐": "N",
	"⛳": "F",
	"❓": "?",
	"🧨": "!",
	"❌": "X",
	"💣": "*",
	"💥": "*",
	"👶": ".",
	"👧": "..",
	"👵": "...",
}

func (a action) icon(s string) string {
	if a.Emoji {
		return s
	}
	boring, ok := emojis[s]
	if !ok {
		return s
	}
	return boring
}

func decode(path string) (*action, error) {
	buf, err := encoding.DecodeString(path)
	if err != nil {
		return nil, fmt.Errorf("base64 decode: %w", err)
	}
	var a action
	err = gob.NewDecoder(bytes.NewBuffer(buf)).Decode(&a)
	if err != nil {
		return nil, fmt.Errorf("gob decode: %w", err)
	}
	return &a, nil
}

const minWidth = 9

func main() {
	opt := gatherOptions()
	if !opt.confirm {
		logrus.Warning("--confirm=false (DRY-RUN): will not write to gcs")
	}
	switch {
	case opt.trace:
		logrus.SetLevel(logrus.TraceLevel)
	case opt.debug:
		logrus.SetLevel(logrus.DebugLevel)
	}

	if c := os.Getenv("APPENGINE_CONFIRM"); c == "true" {
		opt.confirm = true
	}

	if cfg := os.Getenv("APPENGINE_CONFIG"); cfg != "" {
		if err := opt.config.Set(cfg); err != nil {
			logrus.WithError(err).Fatal("Bad config")
		}
	}

	if g := os.Getenv("APPENGINE_GROUP"); g != "" {
		opt.group = g
	}

	seed := time.Now().Unix()
	rand.Seed(seed)

	logrus.WithFields(logrus.Fields{
		"config":  opt.config,
		"confirm": opt.confirm,
		"group":   opt.group,
		"seed":    seed,
	}).Info("Configured")

	logrus.SetReportCaller(true)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	step := func(act *action) {
		act = move(act)
		img := act.Board.render(act)
		w, h := img.Bounds().Dx(), img.Bounds().Dy()
		h += 2
		if w < minWidth {
			w = minWidth
		}
		tool := toolbar(act)
		final := hackimage.New(image.Rect(0, 0, w, h))
		bounds := final.Bounds()
		r := bounds.Sub(bounds.Min).Add(image.Point{})
		draw.Draw(final, r, tool, bounds.Min, draw.Src)
		r = bounds.Sub(bounds.Min).Add(image.Point{0, 2})
		draw.Draw(final, r, img, bounds.Min, draw.Src)

		hackimage.Print(img)
		hackupdater.Update(ctx, "", opt.confirm, final.Cols, nil, opt.config, opt.group)
	}

	step(nil) // clear board on start

	const prefix = "/minesweeper/"
	http.HandleFunc("/minesweeper/", func(w http.ResponseWriter, r *http.Request) {
		p := r.URL.Path[len(prefix):]
		act, err := decode(p)
		if err != nil {
			w.WriteHeader(500)
			fmt.Sprintf("Could not decode: %v", err)
			logrus.WithError(err).Error("failed to decode")
			return
		}
		step(act)
		http.Redirect(w, r, opt.url, http.StatusFound)
	})

	fmt.Println("Listening")
	logrus.Fatal(http.ListenAndServe(":8080", nil))
}

func move(act *action) *action {
	if act == nil {
		act = &action{}
	}
	switch {
	case act.Board == nil || act.New:
		b := generate(settings[act.Difficulty])
		return &action{
			Board:      &b,
			Start:      time.Now().Unix(),
			Difficulty: act.Difficulty,
			Emoji:      act.Emoji,
		}
	case act.Open == nil:
		return act
	default:
		b := *act.Board
		x, y := act.Open.X, act.Open.Y
		cell := b[x][y]
		if act.Flag && !cell.Open {
			switch {
			case cell.Maybe:
				cell.Maybe = false
				cell.Flag = false
			case cell.Flag:
				cell.Maybe = true
			default:
				cell.Flag = true
			}
		} else if !cell.Flag {
			b.open(x, y)
		}
		return act
	}
}
