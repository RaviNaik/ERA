<reponame>firsterblockchain/dynamic-consent
package main

/**
* name    : import
* type	  : import
* comment : 
*/
import (
	"bytes"
	"encoding/json"
	"fmt"
	"strconv"
	"github.com/hyperledger/fabric/core/chaincode/shim"
	pb "github.com/hyperledger/fabric/protos/peer"
)

type SmartContract struct {}

func (s *SmartContract) Init(APIstub shim.ChaincodeStubInterface) pb.Response {
	return shim.Success(nil)
}

func (s *SmartContract) Invoke(APIstub shim.ChaincodeStubInterface) pb.Response {
	function, args := APIstub.GetFunctionAndParameters()
	
	if function == "getAgree" 									{ 	return s.getAgree(APIstub, args)
	} else if function == "setAgree" 							{	return s.setAgree(APIstub, args)
	} else if function == "getAllAgree" 						{	return s.getAllAgree(APIstub)
	} else if function == "getAgreeByWalletId" 					{	return s.getAgreeByWalletId(APIstub, args)
	} else if function == "getAgreeByWalletIdAndAgreeKey" 		{	return s.getAgreeByWalletIdAndAgreeKey(APIstub, args)
	} else if function == "getHistory" 							{	return s.getHistory(APIstub, args)
	} else if function == "setHistory" 							{	return s.setHistory(APIstub, args)
	} else if function == "getAllHistory" 						{	return s.getAllHistory(APIstub)
	} else if function == "getHistoryByWalletId"				{	return s.getHistoryByWalletId(APIstub, args)
	} else if function == "getHistoryByWalletIdAndHistoryType"	{	return s.getHistoryByWalletIdAndHistoryType(APIstub, args)
	} else if function == "updateHistory_Run"					{	return s.updateHistory_Run(APIstub, args)
	}

	fmt.Println("Please check your function : " + function)
	return shim.Error("Unknown function")
}

/**
* name    : main
* type	  : function
* comment : Define main
*/
func main() {

	err := shim.Start(new(SmartContract))
	if err != nil {
		fmt.Printf("Error starting Simple chaincode: %s", err)
	}
}

/**
* name    : Agree
* type	  : structure
* comment : Define Agree structure
*/
type Agree struct {
	UserId			string `json:"userid"`
	AgreeKey 		string `json:"agreekey"`
	Agree	 		string `json:"agree"`
}

/**
* name    : AgreeKey
* type	  : structure
* comment : Define AgreeKey structure
*/
type AgreeKey struct {
	Key string
	Idx int
} 

/**
* name    : HistoryData
* type	  : structure
* comment : Define HistoryData structure
*/
type HistoryData struct {
	UserId			string `json:"userid"`
	HistoryType 	string `json:"historytype"`
	RunType 	 	string `json:"runtype"`
	HashData		string `json:"hashData"`
	CreateDate		string `json:"createdate"`
	RunDate			string `json:"rundate"`
	HashDate		string `json:"hashdate"`
}

/**
* name    : HistoryKey
* type	  : structure
* comment : Define HistoryKey structure
*/
type HistoryKey struct {
	Key string
	Idx int
} 

/**
* name    : generateKey
* type	  : function
* comment : Make key for Agree structure
* @param  	APIstub
* @param  	key
* @return
*/
func generateKey(APIstub shim.ChaincodeStubInterface, key string) []byte {

	var isFirst bool = false

	agreekeyAsBytes, err := APIstub.GetState(key)
	if err != nil {
		fmt.Println(err.Error())
	}

	agreekey := AgreeKey{}
	json.Unmarshal(agreekeyAsBytes, &agreekey)
	var tempIdx string
	tempIdx = strconv.Itoa(agreekey.Idx)
	fmt.Println(agreekey)
	fmt.Println("Key is " + strconv.Itoa(len(agreekey.Key)))
	if len(agreekey.Key) == 0 || agreekey.Key == "" {
		isFirst = true
		agreekey.Key = "MS"
	}
	if !isFirst {
		agreekey.Idx = agreekey.Idx + 1
	}

	fmt.Println("Last agreekey is " + agreekey.Key + " : " + tempIdx)

	returnValueBytes, _ := json.Marshal(agreekey)

	return returnValueBytes
}


/**
* name    : generateKey_History
* type	  : function
* comment : Make key for History structure
* @param  	APIstub
* @param  	key
* @return
*/
func generateKey_History(APIstub shim.ChaincodeStubInterface, key string) []byte {

	var isFirst bool = false

	historykeyAsBytes, err := APIstub.GetState(key)
	if err != nil {
		fmt.Println(err.Error())
	}

	historykey := HistoryKey{}
	json.Unmarshal(historykeyAsBytes, &historykey)
	var tempIdx string
	tempIdx = strconv.Itoa(historykey.Idx)
	fmt.Println(historykey)
	fmt.Println("Key is " + strconv.Itoa(len(historykey.Key)))
	if len(historykey.Key) == 0 || historykey.Key == "" {
		isFirst = true
		historykey.Key = "HS"
	}
	if !isFirst {
		historykey.Idx = historykey.Idx + 1
	}

	fmt.Println("Last historykey is " + historykey.Key + " : " + tempIdx)

	returnValueBytes, _ := json.Marshal(historykey)

	return returnValueBytes
}

/**
* name    : getAgreeByWalletId
* type	  : function
* comment : Search Agree data
* @param  	APIstub
* @param  	args
* @return
*/
func (s *SmartContract) getAgreeByWalletId(APIstub shim.ChaincodeStubInterface, args []string) pb.Response {

	// Check number of arguments 
	if len(args) != 1 {
		return shim.Error("Incorrect number of arguments. Expecting 1")
	}

	// Select query
	queryString := fmt.Sprintf("{\"selector\":{\"userid\":\"%s\"}}", args[0])

	fmt.Printf("queryString:\n%s\n",queryString)

	resultsIterator,err:= APIstub.GetQueryResult(queryString)
	   
	if err != nil {
		return shim.Error(err.Error())
	}

	defer resultsIterator.Close()

	var buffer bytes.Buffer
	buffer.WriteString("[")
	bArrayMemberAlreadyWritten := false

	for resultsIterator.HasNext() {
		queryResponse, err := resultsIterator.Next()
		if err != nil {
			return shim.Error(err.Error())
		}
		
		if bArrayMemberAlreadyWritten == true {
			buffer.WriteString(",")
		}
		buffer.WriteString("{\"Key\":")
		buffer.WriteString("\"")
		buffer.WriteString(queryResponse.Key)
		buffer.WriteString("\"")
		
		buffer.WriteString(", \"Record\":")
		
		buffer.WriteString(string(queryResponse.Value))
		buffer.WriteString("}")
		bArrayMemberAlreadyWritten = true
	}
	buffer.WriteString("]\n")
	return shim.Success(buffer.Bytes())

}

/**
* name    : getAgreeByWalletId
* type	  : function
* comment : 
* @param  	APIstub
* @param  	args
* @return
*/
func (s *SmartContract) getAgreeByWalletIdAndAgreeKey(APIstub shim.ChaincodeStubInterface, args []string) pb.Response {

	if len(args) != 2 {
		return shim.Error("Incorrect number of arguments. Expecting 2")
	}

	queryString := fmt.Sprintf("{\"selector\":{\"userid\":\"%s\",\"agreekey\":\"%s\"}}", args[0],args[1])

	fmt.Printf("queryString:\n%s\n",queryString)

	resultsIterator,err:= APIstub.GetQueryResult(queryString)
	   
	if err != nil {
		return shim.Error(err.Error())
	}

	defer resultsIterator.Close()

	var buffer bytes.Buffer
	buffer.WriteString("[")
	bArrayMemberAlreadyWritten := false

	for resultsIterator.HasNext() {
		queryResponse, err := resultsIterator.Next()
		if err != nil {
			return shim.Error(err.Error())
		}
		
		if bArrayMemberAlreadyWritten == true {
			buffer.WriteString(",")
		}
		buffer.WriteString("{\"Key\":")
		buffer.WriteString("\"")
		buffer.WriteString(queryResponse.Key)
		buffer.WriteString("\"")
		
		buffer.WriteString(", \"Record\":")
		
		buffer.WriteString(string(queryResponse.Value))
		buffer.WriteString("}")
		bArrayMemberAlreadyWritten = true
	}
	buffer.WriteString("]\n")
	return shim.Success(buffer.Bytes())

}


/**
* name    : getAgree
* type	  : function
* comment : 
* @param  	APIstub
* @param  	args
* @return
*/
func (s *SmartContract) getAgree(APIstub shim.ChaincodeStubInterface, args []string) pb.Response {

	agreeAsBytes, err := APIstub.GetState(args[0])
	if err != nil {
		fmt.Println(err.Error())
	}

	agree := Agree{}
	json.Unmarshal(agreeAsBytes, &agree)

	var buffer bytes.Buffer
	buffer.WriteString("[")
	bArrayMemberAlreadyWritten := false

	if bArrayMemberAlreadyWritten == true {
		buffer.WriteString(",")
	}
	buffer.WriteString("{\"Key\":")
	buffer.WriteString("\"")

	buffer.WriteString(agree.AgreeKey)
	buffer.WriteString("\"")

	buffer.WriteString(", \"Agree\":")
	buffer.WriteString("\"")
	buffer.WriteString(agree.Agree)
	buffer.WriteString("\"")

	buffer.WriteString("}")
	bArrayMemberAlreadyWritten = true
	buffer.WriteString("]\n")

	return shim.Success(buffer.Bytes())

}

/**
* name    : setAgree
* type	  : function
* comment : 
* @param  	APIstub
* @param  	args
* @return
*/
func (s *SmartContract) setAgree(APIstub shim.ChaincodeStubInterface, args []string) pb.Response {
	
	if len(args) != 3 {
		return shim.Error("Incorrect number of arguments. Expecting 3")
	}

	queryString := fmt.Sprintf("{\"selector\":{\"userid\":\"%s\",\"agreekey\":\"%s\"}}", args[0],args[1])
	fmt.Printf("queryString:\n%s\n",queryString)

	resultsIterator,err:= APIstub.GetQueryResult(queryString)
	   
	if err != nil {
		return shim.Error(err.Error())
	}

	defer resultsIterator.Close()

	var keyindex = "";

	for resultsIterator.HasNext() {
		queryResponse, err := resultsIterator.Next()
		if err != nil {
			return shim.Error(err.Error())
		}

		keyindex = queryResponse.Key

		break
	}

	if(keyindex != ""){
		agreeAsBytes, err := APIstub.GetState(keyindex)
		if err != nil {
			return shim.Error("chaincode error")
		}
		agree := Agree{}
		json.Unmarshal(agreeAsBytes, &agree)

		agree.Agree = args[2]
		agreeAsBytes, _ = json.Marshal(agree)
		err2 := APIstub.PutState(keyindex, agreeAsBytes)
		if err2 != nil {
			return shim.Error(fmt.Sprintf("Failed to change agree data: %s", keyindex))
		}
		return shim.Success(nil)

	}else{
		var agreekey = AgreeKey{}
		json.Unmarshal(generateKey(APIstub, "latestKey_Agree"), &agreekey)
		keyidx := strconv.Itoa(agreekey.Idx)
		fmt.Println("Key : " + agreekey.Key + ", Idx : " + keyidx)
	
		var agree = Agree{UserId: args[0], AgreeKey: args[1], Agree: args[2]}
		agreeAsJSONBytes, _ := json.Marshal(agree)
	
		var keyString = agreekey.Key + keyidx
		fmt.Println("agreekey is " + keyString)
	
		err := APIstub.PutState(keyString, agreeAsJSONBytes)
		if err != nil {
			return shim.Error(fmt.Sprintf("Failed to record agree catch: %s", agreekey))
		}
	
		agreekeyAsBytes, _ := json.Marshal(agreekey)
		APIstub.PutState("latestKey_Agree", agreekeyAsBytes)
	
		return shim.Success(nil)
	}
	
}

/**
* name    : getAllAgree
* type	  : function
* comment : 
* @param  	APIstub
* @param  	args
* @return
*/
func (s *SmartContract) getAllAgree(APIstub shim.ChaincodeStubInterface) pb.Response {
	
	// Find latestKey
	agreekeyAsBytes, _ := APIstub.GetState("latestKey_Agree")
	agreekey := AgreeKey{}
	json.Unmarshal(agreekeyAsBytes, &agreekey)
	idxStr := strconv.Itoa(agreekey.Idx + 1)

	var startKey = "MS0"
	var endKey = agreekey.Key + idxStr
	fmt.Println(startKey)
	fmt.Println(endKey)

	resultsIter, err := APIstub.GetStateByRange(startKey, endKey)
	if err != nil {
		return shim.Error(err.Error())
	}
	defer resultsIter.Close()
	
	var buffer bytes.Buffer
	buffer.WriteString("[")
	bArrayMemberAlreadyWritten := false
	for resultsIter.HasNext() {
		queryResponse, err := resultsIter.Next()
		if err != nil {
			return shim.Error(err.Error())
		}
		
		if bArrayMemberAlreadyWritten == true {
			buffer.WriteString(",")
		}
		buffer.WriteString("{\"Key\":")
		buffer.WriteString("\"")
		buffer.WriteString(queryResponse.Key)
		buffer.WriteString("\"")
		
		buffer.WriteString(", \"Record\":")
		
		buffer.WriteString(string(queryResponse.Value))
		buffer.WriteString("}")
		bArrayMemberAlreadyWritten = true
	}
	buffer.WriteString("]\n")
	return shim.Success(buffer.Bytes())
}

/**
* name    : getHistoryByWalletId
* type	  : function
* comment : 
* @param  	APIstub
* @param  	args
* @return
*/
func (s *SmartContract) getHistoryByWalletId(APIstub shim.ChaincodeStubInterface, args []string) pb.Response {

	queryString := fmt.Sprintf("{\"selector\":{\"userid\":\"%s\"}}", args[0])

	fmt.Printf("queryString:\n%s\n",queryString)

	resultsIterator,err:= APIstub.GetQueryResult(queryString)
	   
	if err != nil {
		return shim.Error(err.Error())
	}

	defer resultsIterator.Close()

	var buffer bytes.Buffer
	buffer.WriteString("[")
	bArrayMemberAlreadyWritten := false

	for resultsIterator.HasNext() {
		queryResponse, err := resultsIterator.Next()
		if err != nil {
			return shim.Error(err.Error())
		}
		
		if bArrayMemberAlreadyWritten == true {
			buffer.WriteString(",")
		}
		buffer.WriteString("{\"Key\":")
		buffer.WriteString("\"")
		buffer.WriteString(queryResponse.Key)
		buffer.WriteString("\"")
		
		buffer.WriteString(", \"Record\":")
		
		buffer.WriteString(string(queryResponse.Value))
		buffer.WriteString("}")
		bArrayMemberAlreadyWritten = true
	}
	buffer.WriteString("]\n")
	return shim.Success(buffer.Bytes())

}


/**
* name    : getHistoryByWalletIdAndHistoryType
* type	  : function
* comment : 
* @param  	APIstub
* @param  	args
* @return
*/
func (s *SmartContract) getHistoryByWalletIdAndHistoryType(APIstub shim.ChaincodeStubInterface, args []string) pb.Response {

	if len(args) != 2 {
		return shim.Error("Incorrect number of arguments. Expecting 2")
	}

	queryString := fmt.Sprintf("{\"selector\":{\"userid\":\"%s\",\"historytype\":\"%s\",\"runtype\":\"N\"}}", args[0], args[1])

	fmt.Printf("queryString:\n%s\n",queryString)

	resultsIterator,err:= APIstub.GetQueryResult(queryString)
	   
	if err != nil {
		return shim.Error(err.Error())
	}

	defer resultsIterator.Close()

	var buffer bytes.Buffer
	buffer.WriteString("[")
	bArrayMemberAlreadyWritten := false

	for resultsIterator.HasNext() {
		queryResponse, err := resultsIterator.Next()
		if err != nil {
			return shim.Error(err.Error())
		}
		
		if bArrayMemberAlreadyWritten == true {
			buffer.WriteString(",")
		}
		buffer.WriteString("{\"Key\":")
		buffer.WriteString("\"")
		buffer.WriteString(queryResponse.Key)
		buffer.WriteString("\"")
		
		buffer.WriteString(", \"Record\":")
		
		buffer.WriteString(string(queryResponse.Value))
		buffer.WriteString("}")
		bArrayMemberAlreadyWritten = true
	}
	buffer.WriteString("]\n")
	return shim.Success(buffer.Bytes())

}

/**
* name    : getHistory
* type	  : function
* comment : 
* @param  	APIstub
* @param  	args
* @return
*/
func (s *SmartContract) getHistory(APIstub shim.ChaincodeStubInterface, args []string) pb.Response {

	historydataAsBytes, err := APIstub.GetState(args[0])
	if err != nil {
		fmt.Println(err.Error())
	}

	historydata := HistoryData{}
	json.Unmarshal(historydataAsBytes, &historydata)

	var buffer bytes.Buffer
	buffer.WriteString("[")
	bArrayMemberAlreadyWritten := false

	if bArrayMemberAlreadyWritten == true {
		buffer.WriteString(",")
	}
	buffer.WriteString("{\"Type\":")
	buffer.WriteString("\"")

	buffer.WriteString(historydata.HistoryType)
	buffer.WriteString("\"")

	buffer.WriteString(", \"History\":")
	buffer.WriteString("\"")
	buffer.WriteString(historydata.RunType)
	buffer.WriteString("\"")

	buffer.WriteString("}")
	bArrayMemberAlreadyWritten = true
	buffer.WriteString("]\n")

	return shim.Success(buffer.Bytes())

}

/**
* name    : setHistory
* type	  : function
* comment : 
* @param  	APIstub
* @param  	args
* @return
*/
func (s *SmartContract) setHistory(APIstub shim.ChaincodeStubInterface, args []string) pb.Response {

	if len(args) != 2 {
		return shim.Error("Incorrect number of arguments. Expecting 2")
	}

	queryString := fmt.Sprintf("{\"selector\":{\"userid\":\"%s\",\"historytype\":\"%s\",\"runtype\":\"N\"}}", args[0],args[1])
	fmt.Printf("queryString:\n%s\n",queryString)

	resultsIterator,err:= APIstub.GetQueryResult(queryString)
	   
	if err != nil {
		return shim.Error(err.Error())
	}

	defer resultsIterator.Close()

	var keyindex = "";

	for resultsIterator.HasNext() {
		
		queryResponse, err := resultsIterator.Next()

		if err != nil {
			return shim.Error(err.Error())
		}

		keyindex = queryResponse.Key

		break
	}

	if(keyindex != ""){

			return shim.Success(nil)

	}else{

		var historykey = HistoryKey{}
		json.Unmarshal(generateKey_History(APIstub, "latestKey_History"), &historykey)
		keyidx := strconv.Itoa(historykey.Idx)
		fmt.Println("Key : " + historykey.Key + ", Idx : " + keyidx)

		var historydata = HistoryData{UserId: args[0], HistoryType: args[1], RunType: "N", HashData:"", CreateDate:"", RunDate:"", HashDate:""}
		historydataAsJSONBytes, _ := json.Marshal(historydata)
	
		var keyString = historykey.Key + keyidx
		fmt.Println("historykey is " + keyString)
	
		err := APIstub.PutState(keyString, historydataAsJSONBytes)
		if err != nil {
			return shim.Error(fmt.Sprintf("Failed to record historydata catch: %s", historykey))
		}
	
		historykeyAsBytes, _ := json.Marshal(historykey)
		APIstub.PutState("latestKey_History", historykeyAsBytes)
	
		return shim.Success(nil)
	}
	
}

/**
* name    : updateHistory_Run
* type	  : function
* comment : 
* @param  	APIstub
* @param  	args
* @return
*/
func (s *SmartContract) updateHistory_Run(APIstub shim.ChaincodeStubInterface, args []string) pb.Response {
	
	// Check number of arguments
	if len(args) != 3 {
		return shim.Error("Incorrect number of arguments. Expecting 3")
	}

	// Select query
	queryString := fmt.Sprintf("{\"selector\":{\"userid\":\"%s\",\"historytype\":\"%s\",\"runtype\":\"N\"}}", args[0],args[1])
	fmt.Printf("queryString:\n%s\n",queryString)

	// Search
	resultsIterator,err:= APIstub.GetQueryResult(queryString)
	   
	if err != nil {
		return shim.Error(err.Error())
	}

	defer resultsIterator.Close()

	var keyindex = "";

	// Check index
	for resultsIterator.HasNext() {
		queryResponse, err := resultsIterator.Next()

		if err != nil {
			return shim.Error(err.Error())
		}

		keyindex = queryResponse.Key

		break
	}

	if(keyindex != ""){
		if(args[2] == "Y" || args[2] == "N" || args[2] == "E"){
			// ��ȸ
			historydataAsBytes, err := APIstub.GetState(keyindex)
			if err != nil {
				fmt.Println(err.Error())
			}

			historydata := HistoryData{}
			json.Unmarshal(historydataAsBytes, &historydata)
			
			shim.Error(fmt.Sprintf("test data %s, %s",args[2],keyindex))

			historydata.RunType = args[2]
			historydataAsBytes, _ = json.Marshal(historydata)
			err2 := APIstub.PutState(keyindex, historydataAsBytes)
			if err2 != nil {
				return shim.Error(fmt.Sprintf("Failed to change historydata run data: %s", keyindex))
			}
			return shim.Success(nil)

		}else{
			return shim.Error(fmt.Sprintf("input data use Y,N,E"))
		}

	}else{
	
		return shim.Error(fmt.Sprintf("No search data."))
	}
	
}

/**
* name    : getAllHistory
* type	  : function
* comment : 
* @param  	APIstub
* @param  	args
* @return
*/
func (s *SmartContract) getAllHistory(APIstub shim.ChaincodeStubInterface) pb.Response {
	
	// Find latestKey
	historykeyAsBytes, _ := APIstub.GetState("latestKey_History")
	historykey := HistoryKey{}
	json.Unmarshal(historykeyAsBytes, &historykey)
	idxStr := strconv.Itoa(historykey.Idx + 1)

	var startKey = "HS0"
	var endKey = historykey.Key + idxStr
	fmt.Println(startKey)
	fmt.Println(endKey)

	resultsIter, err := APIstub.GetStateByRange(startKey, endKey)
	if err != nil {
		return shim.Error(err.Error())
	}
	defer resultsIter.Close()
	
	var buffer bytes.Buffer
	buffer.WriteString("[")
	bArrayMemberAlreadyWritten := false
	for resultsIter.HasNext() {
		queryResponse, err := resultsIter.Next()
		if err != nil {
			return shim.Error(err.Error())
		}
		
		if bArrayMemberAlreadyWritten == true {
			buffer.WriteString(",")
		}
		buffer.WriteString("{\"Key\":")
		buffer.WriteString("\"")
		buffer.WriteString(queryResponse.Key)
		buffer.WriteString("\"")
		
		buffer.WriteString(", \"Record\":")
		
		buffer.WriteString(string(queryResponse.Value))
		buffer.WriteString("}")
		bArrayMemberAlreadyWritten = true
	}
	buffer.WriteString("]\n")
	return shim.Success(buffer.Bytes())
}