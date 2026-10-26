package weather

import (
	"encoding/json"
	"errors"
	"fmt"
	"github.com/angelokurtis/golang-meetup/internal/log"
	"github.com/angelokurtis/golang-meetup/pkg/location"
	"net/http"
	"os"
)

type OpenWeather struct {
	log        log.Logger
	httpClient *http.Client
	location   location.Locator
}

func NewOpenWeather(log log.Logger, httpClient *http.Client, location location.Locator) *OpenWeather { // Wire's Provider
	return &OpenWeather{log: log, httpClient: httpClient, location: location}
}

func (o *OpenWeather) CheckByCurrentLocation() (*Weather, error) {
	o.log.Info("verificando clima na localidade atual")
	l, err := o.location.WhereAmI()
	if err != nil {
		return nil, err
	}
	return o.CheckByCoord(l.Latitude, l.Longitude)
}

func (o *OpenWeather) CheckByCoord(lat float64, lon float64) (*Weather, error) {
	o.log.Info(fmt.Sprintf("verificando clima atual nas coordenadas de latitude %.2f e longitude %.2f", lat, lon))

	ak := os.Getenv("OPEN_WEATHER_API_ACCESS_KEY")
	r, err := o.httpClient.Get(fmt.Sprintf("http://api.openweathermap.org/data/2.5/weather?lat=%f&lon=%f&APPID=%s&units=metric&lang=pt", lat, lon, ak))
	if err != nil {
		return nil, err
	}
	if r.StatusCode == http.StatusOK {
		b := r.Body
		defer b.Close()
		w := &OpenWeatherResponse{}
		err = json.NewDecoder(b).Decode(w)
		if err != nil {
			return nil, err
		}
		weathers := w.Weather
		if len(weathers) < 1 {
			return nil, errors.New("weathers is empty")
		}
		weather := weathers[0]
		o.log.Info("clima atual encontrado")
		return &Weather{
			Description: weather.Description,
			Temp:        w.Main.Temp,
		}, nil
	}
	return nil, errors.New(fmt.Sprintf("OpenWeather responded unexpectedly (HTTP code %d)", r.StatusCode))
}

type OpenWeatherResponse struct {
	Weather []struct {
		Main        string `json:"main"`
		Description string `json:"description"`
	} `json:"weather"`
	Main struct {
		Temp float64 `json:"temp"`
	} `json:"main"`
}
