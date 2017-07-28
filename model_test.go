package mxnet

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	rice "github.com/GeertJohan/go.rice"
	"github.com/k0kubun/pp"
	"github.com/rai-project/dlframework"
	"github.com/stretchr/testify/assert"
	yaml "gopkg.in/yaml.v2"
)

var (
	builtinModelsBox = rice.MustFindBox("builtin_models")
)

func TestUnmarshalModel(t *testing.T) {
	builtinModelsBox.Walk(".", func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}
		var model dlframework.ModelManifest
		data := builtinModelsBox.MustBytes(path)
		err = yaml.Unmarshal(data, &model)
		assert.NoError(t, err)
		assert.NotEmpty(t, model)
		assert.NoError(t, model.Validate())
		if false {
			pp.Println(model)
		}
		return err
	})

}

func TestModelRegistration(t *testing.T) {
	builtinModelsBox.Walk(".", func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}
		if ext := filepath.Ext(path); ext != ".yml" && ext != ".yaml" {
			return nil
		}
		var model dlframework.ModelManifest
		data := builtinModelsBox.MustBytes(path)
		err = yaml.Unmarshal(data, &model)
		if assert.NoError(t, err) {
			return err
		}
		if assert.NotEmpty(t, model) {
			return errors.New("empty model")
		}

		name, err := model.CanonicalName()
		assert.NoError(t, err)
		assert.NotEmpty(t, name)

		m, err := dlframework.FindModel(name)
		assert.NoError(t, err)
		assert.NotEmpty(t, m)
		// assert.EqualValues(t, model, m)
		// pp.Println(m)
		// pp.Println(model)

		if false {
			pp.Println(model)
		}
		return err
	})
	models, err := dlframework.Models()
	assert.NoError(t, err)
	assert.NotEmpty(t, models)
}
