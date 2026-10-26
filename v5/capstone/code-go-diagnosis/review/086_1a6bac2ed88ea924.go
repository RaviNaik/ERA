<reponame>NeilBetham/elements
package radios

import (
  "log"
  "reflect"
  "sort"
  "time"
  "periph.io/x/periph/conn/spi"
  "periph.io/x/periph/conn/spi/spireg"
  "periph.io/x/periph/conn/gpio"
  "periph.io/x/periph/conn/gpio/gpioreg"
)

var regAddrs = map[string]uint8{
  "opMode": 0x01,
  "dataModulation": 0x02,
  "bitRateMsb": 0x03,
  "bitRateLsb": 0x04,
  "freqDevMsb": 0x05,
  "freqDevLsb": 0x06,
  "carrierFreqMsb": 0x07,
  "carrierFreqMid": 0x08,
  "carrierFreqLsb": 0x09,
  "lnaConfig": 0x18,
  "rxBwFiltCont": 0x19,
  "afcBwFiltCont": 0x1a,
  "afcFeiContStat": 0x1e,
  "afcValMsb": 0x1f,
  "afcValLsb": 0x20,
  "feiValMsb": 0x21,
  "feiValLsb": 0x22,
  "rssiConf": 0x23,
  "rssiValue": 0x24,
  "dioMapping": 0x25,
  "irqFlags1": 0x27,
  "irqFlags2": 0x28,
  "rssiThresh": 0x29,
  "preambleMsb": 0x2c,
  "preambleLsb": 0x2d,
  "syncConfig": 0x2e,
  "syncVal1": 0x2f,
  "syncval2": 0x30,
  "packetConfig1": 0x37,
  "payloadLength": 0x38,
  "packetConfig2": 0x3d,
  "testDagc": 0x6f,
}


type rfm69Regs struct {
  opMode uint8
  dataModulation uint8
  bitRateMsb uint8
  bitRateLsb uint8
  freqDevMsb uint8
  freqDevLsb uint8
  carrierFreqMsb uint8
  carrierFreqMid uint8
  carrierFreqLsb uint8
  lnaConfig uint8
  rxBwFiltCont uint8
  afcBwFiltCont uint8
  afcFeiContStat uint8
  rssiConf uint8
  dioMapping uint8
  irqFlags2 uint8
  rssiThresh uint8
  preambleMsb uint8
  preambleLsb uint8
  syncConfig uint8
  syncVal1 uint8
  syncval2 uint8
  packetConfig1 uint8
  payloadLength uint8
  packetConfig2 uint8
  testDagc uint8
}


func newRFM69Regs() (r rfm69Regs){
  r.opMode = (0x01 << 2) // Standby mode default
  r.dataModulation = (2 << 0) // Gaussian filter, BT= 0.5

  // 19200 bit rate
  r.bitRateMsb = 0x06
  r.bitRateLsb = 0x83

  // 9.9 kHz RX freq deviation
  r.freqDevMsb = 0x00
  r.freqDevLsb = 0x9c

  // Low Noise Amp Config
  // Auto select gain dna 50 impedance input
  r.lnaConfig = 0

  // RX Bandwidth
  // DCC Freq = 2, RX BW Mant = 20, RX BW Mant = 4
  r.rxBwFiltCont = (2 << 5) | (1 << 3) | (4 << 0)

  // AFC BW
  // DCC Freq = 2, RX BW Mant = 20, RX BW Mant = 3
  r.afcBwFiltCont = (2 << 5) | (1 << 3) | (3 << 0)

  // AFC/FEI Settings
  // AutoClear on, AutoOn On
  r.afcFeiContStat = (1 << 3) | (1 << 2)

  // DIO Mapping
  // D0 RX Payload Ready interrupt
  r.dioMapping = (1 << 6)

  // IRQ Flags
  // Clears fifo overrun flag
  r.irqFlags2 = (1 << 4)

  // RSSI Threshold
  r.rssiThresh = 150

  // Preambles
  r.preambleMsb = 0
  r.preambleLsb = 4

  // Sync Config
  // 2 sync bytes, allow two errors in sync bytes
  r.syncConfig = (1 << 7) | (1 << 3) | (2 << 0)

  // Sync Values
  r.syncVal1 = 0xcb
  r.syncval2 = 0x89

  // Packet Config
  // 2 Bit Inter Packet Delay and AutoRxRestartOn On
  r.packetConfig1 = 0
  r.packetConfig2 = (2 << 4) | (1 << 1)

  r.payloadLength = 10

  // Test DAGC Improved No Low Beta On
  r.testDagc = 0x30
  return
}

// RFM69 Handles communication and state for the RFM69 wireless radio
type RFM69 struct {
  port spi.Port
  freq  int32
  conn spi.Conn
  config rfm69Regs
  resetPin gpio.PinIO
  interruptPin gpio.PinIO

  recvBytes []byte
}

// NewRFM69 sets up a new RFM69 class
func NewRFM69(port string, resetPin string, interruptPin string) (r RFM69, err error) {
  p, openErr := spireg.Open(port)
  if openErr != nil {
    log.Printf("error: open spi port failed")
    err = openErr
    return
  }

  r.port = p
  r.freq = 915000000
  r.recvBytes = make([]byte, 1)
  r.config = newRFM69Regs()
  r.resetPin = gpioreg.ByName(resetPin)
  r.interruptPin = gpioreg.ByName(interruptPin)
  r.interruptPin.In(gpio.PullDown, gpio.RisingEdge)

  r.Reset()

  err = r.init()
  return
}

// SetFreq Sets the carrier freq of the RFM69
func (r *RFM69) SetFreq(freq uint32) (err error){
  r.config.carrierFreqLsb = uint8(freq / 61)
  r.config.carrierFreqMid = uint8((freq / 61) >> 8)
  r.config.carrierFreqMsb = uint8((freq / 61) >> 16)
  err = r.syncRegs()
  return
}

func (r *RFM69) setRxMode() (err error){
  r.config.opMode = (0x04 << 2) // Put the chip into RX mode
  err = r.writeReg(regAddrs["opMode"], r.config.opMode)
  return
}

func (r *RFM69) setStdbyMode() (err error){
  r.config.opMode = (0x01 << 2) // Put the chip into standby mode
  err = r.writeReg(regAddrs["opMode"], r.config.opMode)
  return
}

func(r *RFM69) readFifo() (data []uint8, err error){
  bytesToSend := make([]byte, r.config.payloadLength - 1)
  bytesReceived := make([]byte, len(bytesToSend))

  if err = r.conn.Tx(bytesToSend, bytesReceived); err != nil{
    return
  }

  data = bytesReceived[1:]
  return
}

// ReceiveData Waits for a payload to be ready in the
func (r *RFM69) ReceiveData(timeout time.Duration) (pkt Packet, timedout bool, err error){
  r.setRxMode()
  intRecv := r.interruptPin.WaitForEdge(timeout)
  rssi, _ := r.ReadRSSI(false)
  freqErr, _ := r.ReadFreqErr()
  r.setStdbyMode()
  if intRecv {
    timedout = false
  } else {
    timedout = true
    return
  }

  data, _ := r.readFifo()

  pkt.Data = data
  pkt.Freq = int(uint(r.config.carrierFreqLsb) | (uint(r.config.carrierFreqMid) << 8) | (uint(r.config.carrierFreqMsb) << 16)) * 61
  pkt.FreqErr = freqErr
  pkt.Rssi = rssi
  return
}

// DumpRegs dumps the current register settings in the RFM69
func (r *RFM69) DumpRegs() (err error){
  reverseMap := make(map[uint8]string)
  for name, addr := range regAddrs {
    reverseMap[addr] = name
  }

  var keys []int
  for addr := range reverseMap {
     keys = append(keys, int(addr))
  }
  sort.Ints(keys)

  for _, addr := range keys {
    val, _ := r.readReg(uint8(addr))
    log.Printf("%30s | 0x%.2x | %.8b\n", reverseMap[uint8(addr)], val, val)
  }
  return
}

func (r *RFM69) init() (err error){
  // Setup the SPI connection with 5MHz baud, CPOL=0, CPHA=0, and 8 bit bytes
  conn, err := r.port.Connect(5000000, spi.Mode0, 8)
  if err != nil {
    return err
  }
  r.conn = conn

  r.syncRegs()
  return
}

func (r *RFM69) syncRegs() (err error) {
  regValRef := reflect.ValueOf(r.config)
  regTypeRef := reflect.TypeOf(r.config)
  for i := 0; i < regValRef.NumField(); i++ {
    regValue := uint8(regValRef.Field(i).Uint())
    regAddr := regAddrs[regTypeRef.Field(i).Name]
    err = r.writeReg(regAddr, regValue)
    if err != nil {
      break
    }
  }
  return
}

func (r *RFM69) writeReg(addr, val uint8) (err error){
  bytesToSend := []byte{addr | 0x80, val}
  bytesReceived := make([]byte, len(bytesToSend))
  if err = r.conn.Tx(bytesToSend, bytesReceived); err != nil{
    return err
  }
  return
}

func (r *RFM69) readReg(addr uint8) (b byte, err error){
  bytesToSend := []byte{addr & 0x7f, 0x00}
  bytesReceived := make([]byte, len(bytesToSend))
  if err = r.conn.Tx(bytesToSend, bytesReceived); err != nil{
    return 0x00, err
  }
  return bytesReceived[1], nil
}

// ReadRSSI starts an RSSI reading
func (r *RFM69) ReadRSSI(manualStart bool) (rssiVal float64, err error){
  if manualStart{
    err = r.writeReg(regAddrs["rssiConf"], 0x01)

    rssiConfReg, _ := r.readReg(regAddrs["rssiConf"])
    for (rssiConfReg & 0x02) == 0 {
      rssiConfReg, err = r.readReg(regAddrs["rssiConf"])
      if err != nil {
        return
      }
    }
  }

  rssi, err := r.readReg(regAddrs["rssiValue"])
  rssiVal = (float64(rssi) / 2.0) * -1
  return
}

// ReadFreqErr reads the current frequency correction
func (r *RFM69) ReadFreqErr() (freqErr int, err error){
  feiMsb, _ := r.readReg(regAddrs["feiValMsb"])
  feiLsb, _ := r.readReg(regAddrs["feiValLsb"])
  freqErr = int((int16(feiMsb) << 8) | int16(feiLsb)) * 61
  return
}

// Reset resets the RFM69 module
func (r *RFM69) Reset(){
  r.resetPin.Out(gpio.High)
  time.Sleep(5 * time.Millisecond)
  r.resetPin.Out(gpio.Low)
  time.Sleep(50 * time.Millisecond)
}
