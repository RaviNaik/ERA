// Copyright 2019 <NAME>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"flag"
	"fmt"
	"net/http"
	"os"
	"path"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/prometheus/common/log"
	"github.com/prometheus/common/version"

	"calendar_exporter/calendar"
)

func main() {
	var (
		printVersion  = flag.Bool("version", false, "Print the version of the exporter and exit")
		listenAddress = flag.String("web.listen-address", ":9942", "The address to listen on for HTTP requests.")
		metricsPath   = flag.String("web.telemetry-path", "/metrics", "The address to listen on for exporter HTTP requests.")
		calendarPath  = flag.String("web.calendar-path", "/calendar", "The address to listen on for calendar HTTP requests.")
	)
	flag.Parse()

	if *printVersion {
		fmt.Fprintf(os.Stderr, "%s\n", version.Print("calendar_exporter"))
		os.Exit(0)
	}

	calendarList := calendar.NewCalendarList()

	log.Infoln("Starting calendar_exporter", version.Info())
	log.Infoln("Build context", version.BuildContext())

	actualMetricsPath := path.Clean("/" + *metricsPath)
	actualCalendarPath := path.Clean("/"+*calendarPath) + "/"

	http.HandleFunc("/", IndexHandler(actualMetricsPath, actualCalendarPath))
	http.Handle(actualMetricsPath, promhttp.Handler())
	http.Handle(actualCalendarPath, http.StripPrefix(actualCalendarPath, calendarList.Handler()))

	// run exporter
	log.Infoln("Listening on", *listenAddress)
	server := &http.Server{Addr: *listenAddress}
	if err := server.ListenAndServe(); err != nil {
		log.Fatal(err)
	}
}

// IndexHandler returns a handler for root display
func IndexHandler(metricsPath string, calendarPath string) http.HandlerFunc {
	indexHTML := `<html>
  <head><title>Calendar Exporter</title></head>
  <body>
    <h1>Calendar Exporter</h1>
    <p><a href="%s">Metrics</a></p>
    <h1>Calendar Exporter</h1>
    <p><a href="%s">Calendars details</a></p>
  </body>
</html>
`
	index := []byte(fmt.Sprintf(indexHTML, metricsPath, calendarPath))

	return func(w http.ResponseWriter, r *http.Request) {
		w.Write(index)
	}
}
