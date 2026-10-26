<filename>util/util.go
package util

import (
	"encoding/binary"
	"math/rand"
	"net"
	"strconv"
	"strings"
	"time"

	"github.com/jinzhu/gorm"
	"github.com/labstack/echo"
	"github.com/manmanxing/errors"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

func ChangeErr2Grpc(err error) error {
	err = errors.Cause(err)
	if _, ok := status.FromError(err); ok {
		return err
	}
	if err == gorm.ErrRecordNotFound {
		return status.Errorf(codes.NotFound, "非法操作")
	}
	if err != nil {
		return status.Errorf(codes.Internal, err.Error())
	}
	return err
}

func GetClientIP(c echo.Context) string {
	ip := c.Get("client_ip")
	if ip == nil {
		addr := c.Request().RemoteAddr
		ip = strings.Split(addr, ":")[0]
		c.Set("client_ip", ip)
	}
	return ip.(string)
}

func GenerateSpanID(addr string) string {
	strAddr := strings.Split(addr, ":")
	ip := strAddr[0]
	ipLong, _ := Ip2Long(ip)
	times := uint64(time.Now().UnixNano())
	spanId := ((times ^ uint64(ipLong)) << 32) | uint64(rand.Int31())
	return strconv.FormatUint(spanId, 16)
}

func Ip2Long(ip string) (uint32, error) {
	ipAddr, err := net.ResolveIPAddr("ip", ip)
	if err != nil {
		return 0, err
	}
	return binary.BigEndian.Uint32(ipAddr.IP.To4()), nil
}

func GetLocalIP() net.IP {
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		return net.IPv4zero
	}
	for _, a := range addrs {
		if ipnet, ok := a.(*net.IPNet); ok && !ipnet.IP.IsLoopback() {
			if ip := ipnet.IP.To4(); ip != nil {
				return ipnet.IP
			}
		}
	}
	return net.IPv4zero
}

/**
 * 从身份证号中获取生日
 * @params idCard		身份证号
 * @params separator	生日分隔符
 * @return 	生日
 * @return 	错误
 */
func TimeGetBirthday(idCard string, separator string) (birthday string, err error) {
	var (
		year  string
		month string
		day   string
	)
	idCardRune := []rune(idCard)
	if len(idCard) == 18 {
		year = string(idCardRune[6:10])
		month = string(idCardRune[10:12])
		day = string(idCardRune[12:14])
	} else if len(idCard) == 15 {
		year = "19" + string(idCardRune[6:8])
		month = string(idCardRune[8:10])
		day = string(idCardRune[10:12])
	} else {
		return
	}
	birthday = year + separator + month + separator + day
	return
}

func RemoveRepByLoop(slc []int) []int {
	var result []int // 存放结果
	for i := range slc {
		flag := true
		for j := range result {
			if slc[i] == result[j] {
				flag = false // 存在重复元素，标识为false
				break
			}
		}
		if flag { // 标识为false，不添加进结果
			result = append(result, slc[i])
		}
	}
	return result
}

func StringJoin(a []int, split string) string {
	result := ""
	if len(a) <= 0 {
		return result
	}

	for i := range a {
		result += strconv.Itoa(a[i])
		if i < len(a)-1 {
			result += split
		}
	}

	return result
}
