package log

import (
	"fmt"
	"os"
	"time"
)

// DailyFileOutput creates file to output log messages
// which has extension '.log'. The log file
// will be rolled each day with name format prefix_yyyymmdd.log
// and saved inside dirpath. If dirpath not exists, it will be created.
func DailyFileOutput(prefix, dirpath string) (*os.File, error) {
	if _, err := os.Stat(dirpath); os.IsNotExist(err) {
		if err := os.MkdirAll(dirpath, os.ModePerm); err != nil {
			return nil, err
		}
	}

	filename := fmt.Sprintf("%s_%s.log", prefix, time.Now().Format("20060102"))
	file, err := os.OpenFile(filename, os.O_CREATE|os.O_APPEND|os.O_WRONLY, os.ModePerm)
	if err != nil {
		return nil, err
	}

	return file, nil
}
