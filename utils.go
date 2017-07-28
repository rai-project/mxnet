package mxnet

import (
	"encoding/json"
	"errors"
	"fmt"
)

func (e *Graph_NodeEntry) UnmarshalJSON(b []byte) error {
	var s []int64
	if err := json.Unmarshal(b, &s); err != nil {
		return err
	}
	if len(s) < 2 {
		return errors.New("expecting a node entry length >= 2")
	}
	e.NodeId = s[0]
	e.Index = s[1]
	if len(s) == 3 {
		e.Version = s[2]
	}
	return nil
}

func (e *Graph_NodeEntry) MarshalJSON() ([]byte, error) {
	if e.GetVersion() == 0 {
		s := fmt.Sprintf("[\"%d\",\"%d\"]", e.NodeId, e.Index)
		return []byte(s), nil
	}

	s := fmt.Sprintf("[\"%d\",\"%d\",\"%d\"]", e.NodeId, e.Index, e.Version)
	return []byte(s), nil
}
