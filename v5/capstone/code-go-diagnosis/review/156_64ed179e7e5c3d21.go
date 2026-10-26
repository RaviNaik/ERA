package main

import (
	"github.com/abondar24/ZeroMqDemo/clients"
	"github.com/abondar24/ZeroMqDemo/servers"
	"gopkg.in/alecthomas/kingpin.v2"

	"github.com/abondar24/ZeroMqDemo/brokers"
	"github.com/abondar24/ZeroMqDemo/queues"
	"github.com/abondar24/ZeroMqDemo/workers"
	"os"
)

var (
	base               = kingpin.New("base", "ZeroMQ Demo")
	hwSrv              = base.Command("hws", "Run HwServer")
	hwClt              = base.Command("hwc", "Run HwClient")
	ver                = base.Command("-v", "Show ZeroMQ version")
	wuSrv              = base.Command("wus", "Weather Update server")
	wuClt              = base.Command("wuc", "Weather Update Client")
	wuCltArgs          = wuClt.Arg("zipcode", "Location").String()
	taskVent           = base.Command("tv", "Task Ventilator")
	taskWork           = base.Command("tw", "Task Worker")
	taskSink           = base.Command("ts", "Task Sink")
	msReader           = base.Command("msr", "Multi Socket Reader")
	msPoller           = base.Command("msp", "Another Multi Socket Reader")
	rrClient           = base.Command("rrc", "Request Reply Client")
	rrBroker           = base.Command("rrb", "Request Reply Broker")
	rrWorker           = base.Command("rrw", "Request Reply Worker")
	msqQueue           = base.Command("msgq", "Message Queue")
	interrupt          = base.Command("intс", "Interrupt Client")
	mtserver           = base.Command("mts", "Multithreaded hw server")
	mtrelay            = base.Command("mtr", "Multithreaded relay")
	syncpub            = base.Command("spb", "Synchronized publisher")
	syncsub            = base.Command("ssb", "Synchronized subscriber")
	envpub             = base.Command("epb", "Envelope publisher")
	envsub             = base.Command("esb", "Envelope subscriber")
	id                 = base.Command("id", "Identities of sockets")
	rtreq              = base.Command("rtreq", "Router to request")
	llbroker           = base.Command("llb", "Load-Balancing broker")
	llbrokerr          = base.Command("llbr", "Load-Balancing broker with reactor")
	asyncsrv           = base.Command("asys", "Async server")
	peering            = base.Command("peer", "Peering Broker")
	broker             = peering.Arg("broker", "Broker Name").Required().String()
	peers              = peering.Arg("peers", "List of peers").Required().Strings()
	lserver            = base.Command("ls", "Lazy server")
	lclient            = base.Command("lc", "Lazy client")
	rworker            = base.Command("rw", "Reliable worker")
	rqueue             = base.Command("rq", "Reliable queue")
	rrqueue            = base.Command("rrq", "Robust Reliable Queue")
	rorworker          = base.Command("rrw", "Robust reliable Worker")
	mdclient           = base.Command("mdcl", "Majordomo client")
	mdClVerbose        = mdclient.Arg("verbose", "Verbose").Bool()
	mdworker           = base.Command("mdwr", "Majordomo worker")
	mdWrVerbose        = mdworker.Arg("verbose", "Verbose").Bool()
	mdbroker           = base.Command("mdbr", "Majordomo worker")
	mdbrVerbose        = mdbroker.Arg("verbose", "Verbose").Bool()
	mdsearch           = base.Command("mdsrch", "Majordomo search")
	mdsearchVerbose    = mdsearch.Arg("verbose", "Verbose").Bool()
	drclient           = base.Command("drcl", "Disconnected reliable client")
	drclientVerbose    = drclient.Arg("verbose", "Verbose").Bool()
	drbroker           = base.Command("drbr", "Disconnected reliable broker")
	drbrokerVerbose    = drbroker.Arg("verbose", "Verbose").Bool()
	haserver           = base.Command("hasrv", "Highly Available server")
	haserverPrimary    = haserver.Arg("primary", "Primary server").Bool()
	haclient           = base.Command("haclt", "Highly Available client")
	brlServerFail      = base.Command("bsf", "Brokerless failover server")
	brlEndpoint        = brlServerFail.Arg("endpoint", "Server endpoint").String()
	brlClientFail      = base.Command("bcf", "Brokerless failover client")
	brlEndpoints       = brlClientFail.Arg("endpoints", "Server endpoints").Strings()
	brlMsgServerFail   = base.Command("bsmf", "Brokerless failover server sending message to client")
	brlMsgEndpoint     = brlMsgServerFail.Arg("serverEndpoint", "Server endpoint").String()
	brlRepClientFail   = base.Command("bcrf", "Brokerless failover client sending multiple replies to server")
	brlRepEndpoints    = brlRepClientFail.Arg("servEndpoint", "Server endpoint").Strings()
	brlRtServer        = base.Command("bsrtf", "Brokerless failover server with routing")
	brlRtServerPort    = brlRtServer.Arg("port", "Port").String()
	brlRtServerVerbose = brlRtServer.Arg("isVerbose", "Is verbose?").Bool()
	brlRtClient        = base.Command("bcrtf", "Brokerless failover async client")
	pstrace            = base.Command("pst", "Pub-Sub Tracing")
	ptpubl             = base.Command("ptp", "Pathological publisher")
	ptpCache           = ptpubl.Arg("pcache", "Cache required?").Bool()
	ptsub              = base.Command("pts", "Pathological subscriber")
	ptsCache           = ptsub.Arg("scache", "Cache required?").Bool()
	lvcache            = base.Command("lvc", "Last Value Cache")
	slowSubDetection   = base.Command("ssd", "Slow Subscriber detection")
	relPSServer        = base.Command("rpss", "Reliable Pub-Sub Server")
	relPSServerPrimary = relPSServer.Arg("rpssp", "is primary?").Bool()
	relPSClient        = base.Command("rpsc", "Reliable Pub-Sub Client")
	fileTransfer       = base.Command("ft", "File transfer via ZeroMQ")
	filePath           = fileTransfer.Arg("fname", "File path").String()
)

func main() {
	cmd := kingpin.MustParse(base.Parse(os.Args[1:]))

	switch cmd {

	case hwSrv.FullCommand():
		servers.HwServer()

	case hwClt.FullCommand():
		clients.HwClient()

	case ver.FullCommand():
		Version()

	case wuSrv.FullCommand():
		servers.WuServer()

	case wuClt.FullCommand():
		clients.WuClient(*wuCltArgs)

	case taskVent.FullCommand():
		servers.TaskVentilator()

	case taskWork.FullCommand():
		workers.TaskWorker()

	case taskSink.FullCommand():
		clients.TaskSink()

	case msReader.FullCommand():
		clients.MsReader()

	case msPoller.FullCommand():
		clients.MsPoller()

	case rrClient.FullCommand():
		clients.RRclient()

	case rrBroker.FullCommand():
		brokers.RRbroker()

	case rrWorker.FullCommand():
		workers.RequestReplyRworker()

	case msqQueue.FullCommand():
		queues.MsgQueue()

	case interrupt.FullCommand():
		clients.Interrupt()

	case mtserver.FullCommand():
		servers.MTserver()

	case mtrelay.FullCommand():
		MTrelay()

	case syncpub.FullCommand():
		servers.SyncPub()

	case syncsub.FullCommand():
		clients.SyncSub()

	case envpub.FullCommand():
		servers.EnvPub()

	case envsub.FullCommand():
		clients.EnvSub()

	case id.FullCommand():
		Identity()

	case rtreq.FullCommand():
		ReqRouter()

	case llbroker.FullCommand():
		brokers.LoadBalacningBroker()

	case llbrokerr.FullCommand():
		brokers.LoadBalacningReactorBroker()

	case asyncsrv.FullCommand():
		AsyncServer()

	case peering.FullCommand():
		Peering(*broker, *peers...)

	case lserver.FullCommand():
		servers.LazyServer()

	case lclient.FullCommand():
		clients.LazyClient()

	case rworker.FullCommand():
		workers.ReliableWorker()

	case rqueue.FullCommand():
		queues.ReliableQueue()

	case rrqueue.FullCommand():
		queues.RobustReliableQueue()

	case rorworker.FullCommand():
		workers.RobustReliableWorker()

	case mdclient.FullCommand():
		clients.MajordomoClient(*mdClVerbose)

	case mdworker.FullCommand():
		workers.MajordomoWorker(*mdWrVerbose)

	case mdbroker.FullCommand():
		brokers.MajordomoBroker(*mdbrVerbose)

	case mdsearch.FullCommand():
		clients.MajordomoDiscoverySearch(*mdsearchVerbose)

	case drclient.FullCommand():
		clients.DisconnectedReliableClient(*drclientVerbose)

	case drbroker.FullCommand():
		brokers.DisconnectedReliableBroker(*drbrokerVerbose)

	case haserver.FullCommand():
		servers.HAServer(*haserverPrimary)

	case haclient.FullCommand():
		clients.HighlyAvailableClient()

	case brlServerFail.FullCommand():
		servers.BrokerlessFailoverServer(*brlEndpoint)

	case brlClientFail.FullCommand():
		clients.BrokerlessFailoverClient(*brlEndpoints...)

	case brlMsgServerFail.FullCommand():
		servers.BrokerlessFailoverMsgReplyServer(*brlMsgEndpoint)

	case brlRepClientFail.FullCommand():
		clients.BrokerlessFailoverReplyClient(*brlRepEndpoints...)

	case brlRtServer.FullCommand():
		servers.BrokerlessRoutingServer(*brlRtServerPort, *brlRtServerVerbose)

	case brlRtClient.FullCommand():
		clients.BrokerlessFailoverRoutingClient()

	case pstrace.FullCommand():
		PubSubTracing()

	case ptpubl.FullCommand():
		servers.PathologicalPublisher(*ptpCache)

	case ptsub.FullCommand():
		clients.PathologicalSubscriber(*ptsCache)

	case lvcache.FullCommand():
		servers.LastValueCache()

	case slowSubDetection.FullCommand():
		SlowSubscriberDetection()

	case relPSServer.FullCommand():
		servers.ReliablePubSubServer(*relPSServerPrimary)

	case relPSClient.FullCommand():
		clients.ReliablePubSubClient()

	case fileTransfer.FullCommand():
		FileTransfer(*filePath)
	}

	os.Exit(0)
}
