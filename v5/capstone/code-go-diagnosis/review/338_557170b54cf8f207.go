package tailing

import (
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"

	"github.com/dimpogissou/isengard-server/logger"
	"github.com/dimpogissou/isengard-server/observer"
	"github.com/hpcloud/tail"
	"gopkg.in/fsnotify.v1"
)

// Collects file names in provided directory as an array of strings
func getFileNamesInDir(dir string) []string {
	files, err := ioutil.ReadDir(dir)
	if err != nil {
		logger.Error("FailedRetrievingFiles", fmt.Sprintf("Could not get files from directory %s due to -> %s", dir, err))
	}
	paths := []string{}
	for _, f := range files {
		paths = append(paths, f.Name())
	}
	return paths
}

// Starts tailing a file at provided path, from start if whence = 0, from the end if whence = 2
func createTail(path string, whence int) (*tail.Tail, error) {

	logger.Info(fmt.Sprintf("Start tailing file %s", path))

	t, err := tail.TailFile(path, tail.Config{Follow: true, MustExist: true, Location: &tail.SeekInfo{Offset: 0, Whence: whence}, ReOpen: true, Poll: true})

	return t, err
}

// Starts tailing all files in provided directory
func InitTailsFromDir(dir string) []*tail.Tail {

	var files = getFileNamesInDir(dir)
	var tails = make([]*tail.Tail, 0)
	for _, fileName := range files {
		filePath := fmt.Sprintf("%s/%s", dir, fileName)
		t, err := createTail(filePath, io.SeekEnd)
		if err != nil {
			logger.Error("FailedTailingFile", fmt.Sprintf("Could not tail file [%s] due to -> %s", filePath, err))
		} else {
			tails = append(tails, t)
		}
	}
	return tails
}

// Routine tailing a file and sending lines to logs channel
func TailAndPublish(lines chan *tail.Line, publisher observer.Publisher) {
	for line := range lines {
		publisher.Publish(line)
	}
}

// Monitors and tails new files, returns on signal interruption
func TailNewFiles(watcher *fsnotify.Watcher, logsPublisher observer.Publisher, sigChan chan os.Signal) {

	for {
		select {
		case event, ok := <-watcher.Events:
			if !ok {
				logger.CheckErrAndLog(errors.New("Received fatal error from watcher.Events channel"), "WatcherError", "Error occured in filewatching routine")
				return
			}
			if event.Op&fsnotify.Create == fsnotify.Create {
				// Tail new file from beginning of file
				t, err := createTail(event.Name, io.SeekStart)
				defer t.Stop()
				if err != nil {
					logger.CheckErrAndLog(err, "FailedTailingNewFile", fmt.Sprintf("Error occured at tail creation for %s", event.Name))
				} else {
					go TailAndPublish(t.Lines, logsPublisher)
				}
			}
		case err, ok := <-watcher.Errors:
			logger.CheckErrAndLog(err, "ReceivedWatcherError", fmt.Sprintf("Received error from watcher.Errors channel"))
			if !ok {
				logger.CheckErrAndLog(err, "FatalWatcherError", fmt.Sprintf("Received fatal error from watcher.Errors channel"))
				return
			}
		case <-sigChan:
			logger.Info("Termination signal received, running deferred ops and exiting...")
			return
		}
	}

}
