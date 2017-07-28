package mxnet

import (
	"encoding/json"
	"testing"

	rice "github.com/GeertJohan/go.rice"
	"github.com/stretchr/testify/assert"
)

var (
	fixturesBox         = rice.MustFindBox("_fixtures")
	inceptionSymbolJSON = fixturesBox.MustBytes("Inception-BN-symbol.json")
	caffenetSymbolJSON  = fixturesBox.MustBytes("caffenet-symbol.json")
	rn101               = fixturesBox.MustBytes("RN101-5k500-symbol.json")
)

func TestUnmarshalGraph(t *testing.T) {
	var g Graph
	err := json.Unmarshal(rn101, &g)
	assert.NoError(t, err)
	assert.NotEmpty(t, g)

	s, err := json.MarshalIndent(g, "", "  ")
	assert.NoError(t, err)
	assert.NotEmpty(t, s)
	// t.Log(string(s))

	dg, err := g.ToDotGraph()
	assert.NoError(t, err)
	assert.NotNil(t, dg)
	assert.NotEmpty(t, dg)

	t.Log(dg.String())

}
