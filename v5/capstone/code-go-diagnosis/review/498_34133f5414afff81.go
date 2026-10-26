/*
package commonLib

nanomsg를 통해 LOXILIGHT코어에 API를 날리는 함수를 모아놓은 패키지입니다.

헬리오스가 로컬에 켜져있을때 사용해야 작동이 됩니다. 디폴트 서버 주소가 127.0.0.1입니다.

만약 켰는데 안되면 ifconfig lo 를 확인해서 up인지 아닌지 확인 해보세요.
*/
package commonLib

import (
	"encoding/binary"
	"errors"
	"fmt"
	"net"
	"os"
	"strconv"
	"unsafe"

	"github.com/olekukonko/tablewriter"
	"go.nanomsg.org/mangos/v3"
	"go.nanomsg.org/mangos/v3/protocol/req"
	_ "go.nanomsg.org/mangos/v3/transport/all"
)

type Hls_idl_hdr struct {
	version uint8
	cmd     uint8
	length  uint16
	xid     uint32
}

type Hls_idl_byte_hdr struct {
	version uint8
	cmd     uint8
	length  []byte
	xid     []byte
	data    []byte
}

// version
const VERSION = 1

// LOXILIGHT Controller IP주소
var Controller_ip string = "tcp://127.0.0.1:8861"

const (
	LOXILIGHT_PACKET_IN                     = iota + 1 /* Legacy packet in from SDN edge */
	LOXILIGHT_PACKET_OUT                               /* Legacy packet out toward SDN edge */
	LOXILIGHT_VERSION_SHOW                             /* *Version show */
	LOXILIGHT_RUN_CONFIG_SHOW                          /* *Legacy current configuration show */
	LOXILIGHT_START_CONFIG_SHOW                        /* *Legacy start up configuration show */
	LOXILIGHT_INTERFACE_STAT_SHOW_BRIEF                /* *Legacy Interface statistics counter show all */
	LOXILIGHT_INTERFACE_STAT_SHOW_INTERFACE            /* *Legacy Interface statistics counter show with interface name */
	LOXILIGHT_INTERFACE_SHOW_BRIEF                     /* *Legacy Interface information show (MAC, Speed, Port mode etc.)*/
	LOXILIGHT_INTERFACE_SHOW_INTERFACE                 /* *Legacy Interface information show with interface name */
	LOXILIGHT_IP_ADDR_ADD                              /* Legacy Interface address addition */
	LOXILIGHT_IP_ADDR_DEL                              /* Legacy Interface address deletion */
	LOXILIGHT_IPV6_ADDR_ADD                            /* Legacy ipv6 Interface address addition */
	LOXILIGHT_IPV6_ADDR_DEL                            /* Legacy ipv6 Interface address deletion */
	LOXILIGHT_IP_INTERFACE_BRIEF                       /* *Legacy IP information show (IP address, MTU, Proxy arp etc. )*/
	LOXILIGHT_IP_INTERFACE                             /* *Legacy IP information show with interface name */
	LOXILIGHT_RT_ADD                                   /* Legacy route addition */
	LOXILIGHT_RT_DEL                                   /* Legacy route deletion */
	LOXILIGHT_IPV6_RT_ADD                              /* Legacy ipv6 route addition */
	LOXILIGHT_IPV6_RT_DEL                              /* Legacy ipv6 route deletion */
	LOXILIGHT_RT_SHOW_ALL                              /* *Legacy route show both of IPv4 and IPv6*/
	LOXILIGHT_NH_ADD                                   /* Legacy next-hop addition */
	LOXILIGHT_NH_DEL                                   /* Legacy next-hop deletion */
	LOXILIGHT_IPV6_NH_ADD                              /* Legacy next-hop addition */
	LOXILIGHT_IPV6_NH_DEL                              /* Legacy next-hop deletion */
	LOXILIGHT_NH_SHOW_ALL                              /* *Legacy next-hop show both of IPv4 and IPv6 */
	LOXILIGHT_NH_SHOW_INTERFACE                        /* *Legacy next-hop show both of IPv4 and IPv6 with interface name */
	LOXILIGHT_VBR_ADD                                  /* VLAN Bridge Creation/Modification*/
	LOXILIGHT_VBR_DEL                                  /* VLAN Bridge Deletion */
	LOXILIGHT_VBR_SHOW_ALL                             /* *VLAN Bridge Show with brief infomation*/
	LOXILIGHT_VBR_SHOW_ID                              /* *VLAN Bridge Show using vlan id*/
	LOXILIGHT_VBR_PORT_ADD                             /* Port Add to VLAN Bridge*/
	LOXILIGHT_VBR_PORT_DEL                             /* Port Delete from VLAN Bridge */
	LOXILIGHT_VXBR_ADD                                 /* VxLAN Bridge Creation/Modification*/
	LOXILIGHT_VXBR_DEL                                 /* VxLAN Bridge Deletion */
	LOXILIGHT_VXBR_AP_ADD                              /* VxLAN Bridge Access Point add*/
	LOXILIGHT_VXBR_AP_DEL                              /* VxLAN Bridge Access Point del*/
	LOXILIGHT_VXBR_EP_RIP_ADD                          /* VxLAN Bridge Remote IP add*/
	LOXILIGHT_VXBR_EP_RIP_DEL                          /* VxLAN Bridge Remote IP del*/
	LOXILIGHT_VXBR_SHOW_ALL                            /* VxLAN Bridge Show */
	LOXILIGHT_VXBR_SHOW_ID                             /* VxLAN Bridge Show using vxlan id*/
	LOXILIGHT_VTEP_SHOW_ALL                            /* VxLAN Tunnel Endpoint Show */
	LOXILIGHT_L2FDB_ADD                                /* MAC entry add in Legacy L2 Table */
	LOXILIGHT_L2FDB_DEL                                /* MAC entry del from Legacy L2 Table */
	LOXILIGHT_VXFDB_ADD                                /* MAC entry add in Legacy L2 Table */
	LOXILIGHT_VXFDB_DEL                                /* MAC entry del from Legacy L2 Table */
	LOXILIGHT_L2FDB_SHOW_ALL                           /* MAC entry show from Legacy L2 Table */
	LOXILIGHT_L2FDB_SHOW_VLAN                          /* MAC entry show from Legacy L2 Table for vlan id */
	LOXILIGHT_VXFDB_SHOW_ALL                           /* MAC entry show from Legacy L2 Table */
	LOXILIGHT_VXFDB_SHOW_VXLAN                         /* MAC entry show from Legacy L2 Table for vxlan id */
	LOXILIGHT_LAG_INTF_ADD                             /* LAG Interface Creation/Modification*/
	LOXILIGHT_LAG_INTF_DEL                             /* LAG Interface Deletion */
	LOXILIGHT_LAG_PORT_ADD                             /* Port Add to LAG Interface*/
	LOXILIGHT_LAG_PORT_DEL                             /* Port Delete from LAG Interface */
	LOXILIGHT_LAG_MODE_SET                             /* LAG Interface mode set*/
	LOXILIGHT_IPSEC_POLICY_ADD                         /* IPSEC policy add */
	LOXILIGHT_IPSEC_POLICY_DEL                         /* IPSEC policy del */
	LOXILIGHT_IPSEC_POLICY_SHOW                        /* IPSEC policy show */
	LOXILIGHT_STP_STATE                                /* STP state from Port state in bridge*/
	LOXILIGHT_SDN_VIRT_IF_ADD                          /* Virtual interface addition from SDN core */
	LOXILIGHT_SDN_VIRT_IF_DEL                          /* Virtual interface deletion from SDN core */
	LOXILIGHT_SDN_VIRT_IF_UPDATE                       /* Virtual interface update */
	LOXILIGHT_VIF_SHOW_ALL                             /* Virtual interface show all */
	LOXILIGHT_VIF_SHOW_NAME                            /* Virtual interface show by name */
	LOXILIGHT_MIRROR_ADD                               /* MIRROR add */
	LOXILIGHT_MIRROR_DEL                               /* MIRROR del */
	LOXILIGHT_MIRROR_PORT_ADD                          /* MIRROR inbound port add */
	LOXILIGHT_MIRROR_PORT_DEL                          /* MIRROR inbound port del */
	LOXILIGHT_MIRROR_SHOW                              /* MIRROR show */
	LOXILIGHT_ACL_ADD                                  /* ACL add */
	LOXILIGHT_ACL_DEL                                  /* ACL del */
	LOXILIGHT_ACL_SHOW                                 /* ACL show */
	LOXILIGHT_CT_SHOW                                  /* Connection tracking show */
	LOXILIGHT_APPFLOW_SHOW                             /* Cache flow show */
	LOXILIGHT_SERVICE_ECHO
	LOXILIGHT_SERVICE_SUCCESS
	LOXILIGHT_SERVICE_ERROR
)

func makeTable(title []string, data [][]string) {
	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader(title)
	table.SetBorders(tablewriter.Border{Left: true, Top: false, Right: true, Bottom: false})
	table.SetCenterSeparator("|")
	table.AppendBulk(data) // Add Bulk Data
	table.Render()
}

func GetConnection(url string) (mangos.Socket, error) {
	var sock mangos.Socket
	var err error

	if sock, err = req.NewSocket(); err != nil {
		die("can't get new req socket: %s", err.Error())
	}
	if err = sock.Dial(url); err != nil {
		die("can't dial on req socket: %s", err.Error())
	}

	return sock, err
}

func GetConnectionCheck(url string) error {
	var sock mangos.Socket
	var err error

	if sock, err = req.NewSocket(); err != nil {
		die("can't get new req socket: %s", err.Error())
	}
	if err = sock.Dial(url); err != nil {
		die("can't dial on req socket: %s", err.Error())
	}

	sock.Close()
	return err
}

func ChangeControllerIp(url string) {
	Controller_ip = url
}

func ChangeHexIpToDec(hexip string) string {
	one, err := strconv.ParseInt(hexip[2:4], 16, 64)
	if err != nil {
		fmt.Printf("Conversion failed: %s\n", err)
	}
	two, err := strconv.ParseInt(hexip[4:6], 16, 64)
	if err != nil {
		fmt.Printf("Conversion failed: %s\n", err)
	}
	three, err := strconv.ParseInt(hexip[6:8], 16, 64)
	if err != nil {
		fmt.Printf("Conversion failed: %s\n", err)
	}
	four, err := strconv.ParseInt(hexip[8:10], 16, 64)
	if err != nil {
		fmt.Printf("Conversion failed: %s\n", err)
	}
	return fmt.Sprintf("%d.%d.%d.%d", one, two, three, four)
}

func SendMessage(sock mangos.Socket, txt_msg []byte) string {
	var err error
	var msg []byte

	if err = sock.Send(txt_msg); err != nil {
		die("can't send message on push socket: %s", err.Error())
	}
	if msg, err = sock.Recv(); err != nil {
		die("can't receive date: %s", err.Error())
	}
	sock.Close()
	return string(msg[8:])
}

func CloseConnection(sock mangos.Socket) {
	sock.Close()
}

func MakeHeader(command uint8, msg string) (Hls_idl_hdr, string) {
	var header Hls_idl_hdr
	ver := uint8(VERSION)
	cmd := uint8(command)
	length := uint16(len(msg) + int(unsafe.Sizeof(header)))
	xid := uint32(0)
	header.version = ver
	header.cmd = cmd
	header.length = length
	header.xid = xid
	//fmt.Printf("msg length = %d, header length = %d", len(msg), int(unsafe.Sizeof(header)))
	ret := header.String()
	return header, ret
}

func MakeMessage(command uint8, msg string) (Hls_idl_byte_hdr, []byte) {
	var header Hls_idl_byte_hdr
	var tmp_hdr Hls_idl_hdr
	ver := uint8(VERSION)
	cmd := uint8(command)
	length := htons(uint16(len(msg) + int(unsafe.Sizeof(tmp_hdr))))
	xid := htonl(0)

	header.version = ver
	header.cmd = cmd
	header.length = length
	header.xid = xid
	ret := header.Byte()
	ret = append(ret, []byte(msg)...)
	//fmt.Print("Make a message : ")
	//fmt.Println(ret)
	return header, ret
}

func ShowVersion() {
	var sock mangos.Socket
	sock, err := GetConnection(Controller_ip)
	if err != nil {
		fmt.Println("Please check your Core APP and CLI network status")
		return
	}
	// make data format
	msg := ""
	cmd := uint8(LOXILIGHT_VERSION_SHOW)
	_, hdr := MakeMessage(cmd, msg)

	// send msg and return value
	res := SendMessage(sock, hdr)
	fmt.Printf("Receive %s\n", string(res))

}

func GetRunningConfig() (string, error) {
	fdb, err := GetFdbConfig()
	if err != nil {
		return "Get FDB Fail", err
	}
	vxfdb, err := GetVxFdbConfig()
	if err != nil {
		return "Get VXFDB Fail", err
	}
	route, err := GetIpRouteConfig()
	if err != nil {
		return "Get Route Fail", err
	}
	neigh, err := GetIpNeighborConfig()
	if err != nil {
		return "Get Neigh Fail", err
	}
	ipsec, err := GetIpsecConfig()
	if err != nil {
		return "Get IPSEC Fail", err
	}
	vlan, err := GetVlanBriefData()
	if err != nil {
		return "Get VLAN Fail", err
	}
	vxlan, err := GetVxlanBriefConfig()
	if err != nil {
		return "Get VXLAN Fail", err
	}

	ret := fmt.Sprintf(`-FDB-\n%s\n-FDBEND-\n
						-VXFDB-\n%s\n-VXFDBEND-\n
						-ROUTE-\n%s\n-ROUTEEND-\n
						-NEIGHBOR-\n%s\n-NEIGHBOREND-\n
						-IPSEC-\n%s\n-IPSECEND-\n
						-VLAN-\n%s\n-VLANEND-\n
						-VXLAN-\n%s\n-VXLANEND-\n
						`, fdb, vxfdb, route, neigh, ipsec, vlan, vxlan)
	return ret, err
}

func ShowStartupConfig() {
	var sock mangos.Socket
	sock, err := GetConnection(Controller_ip)
	if err != nil {
		fmt.Println("Please check your Core APP and CLI network status")
		return
	}
	msg := ""
	cmd := uint8(LOXILIGHT_START_CONFIG_SHOW)
	_, hdr := MakeMessage(cmd, msg)

	// send msg and return value
	res := SendMessage(sock, hdr)
	fmt.Printf("Receive %s\n", string(res))

}

func (hdr Hls_idl_hdr) String() string {
	return string(hdr.version) + string(hdr.cmd) + string(hdr.length) + string(hdr.xid)
}

func (hdr Hls_idl_byte_hdr) Byte() []byte {
	ret := []byte(string(hdr.version) + string(hdr.cmd))
	ret = append(ret, hdr.length...)
	ret = append(ret, hdr.xid...)
	return ret
}

func IsValidIP(ipstr string) bool {
	ip := net.ParseIP(ipstr)
	return (ip != nil) && (ip.To4() != nil)
}

func IsV4(ipstr string) bool {
	ip := net.ParseIP(ipstr)
	return (ip.To4() != nil)
}

// IsHexID 는 "0x"로 시작하는것에 대해서만 검사합니다.
func IsHexID(hex_string string) bool {
	_, err := strconv.ParseUint(hex_string[2:], 16, 64)
	return err == nil
}

func ValidateIsV4(ipstr string) (string, bool) {
	// If v4 IP, return true else false
	ip := net.ParseIP(ipstr)                       // check IP validation
	return ipstr, (ip != nil) && (ip.To4() == nil) // check v4,v6
}

func ValidateIPBoth(ipstr string) (string, bool) {
	ip := net.ParseIP(ipstr)
	return ipstr, ip == nil
}

func ValidateVlanID(vlan_id_str string) (vlanID int, err error) {
	vlanID, err = strconv.Atoi(vlan_id_str)
	if err == nil {
		if vlanID < 2 || vlanID > 4094 {
			err = errors.New("vlanID out of range " + vlan_id_str)
		}
	}
	return
}

func ValidateVxlanID(vlan_id_str string) (vxlanID int, err error) {
	vxlanID, err = strconv.Atoi(vlan_id_str)
	if err == nil {
		if vxlanID < 2 || vxlanID > 16777215 {
			err = errors.New("vxlanID out of range " + vlan_id_str)
		}
	}
	return
}

func ParseIPBothPrefix(ipprefix string) (ipstr string, length int, err error) {
	ip, net, err := net.ParseCIDR(ipprefix)
	if err != nil {
		return
	}

	ipstr = ip.String()
	length, _ = net.Mask.Size()

	return
}

func die(format string, v ...interface{}) {
	fmt.Fprintln(os.Stderr, fmt.Sprintf(format, v...))
}

func htonl(val uint32) []byte {
	bytes := make([]byte, 4)
	binary.BigEndian.PutUint32(bytes, val)
	return bytes
}

func htons(val uint16) []byte {
	bytes := make([]byte, 2)
	binary.BigEndian.PutUint16(bytes, val)
	return bytes
}

func ntohl(buf []byte) uint32 {
	return binary.BigEndian.Uint32(buf)
}

func ntohs(buf []byte) uint16 {
	return binary.BigEndian.Uint16(buf)
}
