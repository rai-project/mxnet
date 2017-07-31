package main

import (
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strconv"

	"github.com/Unknwon/com"
	"github.com/facebookgo/freeport"
	"github.com/fatih/color"
	"github.com/pkg/errors"
	"github.com/rai-project/config"
	"github.com/rai-project/mxnet/agent"
	"github.com/rai-project/utils"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var (
	isDebug   bool
	isVerbose bool
	local     bool
	appSecret string
	cfgFile   string
)

func freePort() (string, error) {
	port, err := freeport.Get()
	if err != nil {
		return "", err
	}
	return strconv.Itoa(port), nil
}

func getAddress(port string) (string, error) {
	var address string
	var err error
	if local {
		address, err = utils.GetLocalIp()
	} else {
		address, err = utils.GetExternalIp()
	}
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("%s:%s", address, port), nil
}

// RootCmd represents the base command when called without any subcommands
var RootCmd = &cobra.Command{
	Use:   "mxnet-agent",
	Short: "Runs the carml MXNet agent",
	RunE: func(cmd *cobra.Command, args []string) error {
		port, found := os.LookupEnv("PORT")
		if !found {
			p, err := freePort()
			if err != nil {
				return err
			}
			port = p
		}

		address, err := getAddress(port)
		if err != nil {
			return err
		}

		registeryServer, err := agent.RegisterRegistryServer()
		if err != nil {
			return err
		}

		predictorServer, err := agent.RegisterPredictorServer(address)
		if err != nil {
			return err
		}

		lis, err := net.Listen("tcp", address)
		if err != nil {
			return errors.Wrapf(err, "failed to listen on ip %s", address)
		}

		// log.Debug("mxnet service is listening on %s", address)

		defer registeryServer.GracefulStop()
		defer predictorServer.GracefulStop()

		go registeryServer.Serve(lis)
		predictorServer.Serve(lis)
		return nil
	},
}

func init() {
	cobra.OnInitialize(initConfig)

	RootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", "config file (default is $HOME/.carml_config.yaml)")
	RootCmd.PersistentFlags().BoolVarP(&isVerbose, "verbose", "v", false, "Toggle verbose mode.")
	RootCmd.PersistentFlags().BoolVarP(&isDebug, "debug", "d", false, "Toggle debug mode.")
	RootCmd.PersistentFlags().StringVarP(&appSecret, "secret", "s", "", "The application secret.")
	RootCmd.PersistentFlags().BoolVarP(&local, "local", "l", false, "Listen on local address.")

	// Cobra also supports local flags, which will only run
	// when this action is called directly.
	RootCmd.Flags().BoolP("toggle", "t", false, "Help message for toggle")

	viper.BindPFlag("app.secret", RootCmd.PersistentFlags().Lookup("secret"))
	viper.BindPFlag("app.debug", RootCmd.PersistentFlags().Lookup("debug"))
	viper.BindPFlag("app.verbose", RootCmd.PersistentFlags().Lookup("verbose"))
}

// initConfig reads in config file and ENV variables if set.
func initConfig() {

	color.NoColor = false
	opts := []config.Option{
		config.AppName("carml"),
		config.ColorMode(true),
	}
	if com.IsFile(cfgFile) {
		if c, err := filepath.Abs(cfgFile); err == nil {
			cfgFile = c
		}
		opts = append(opts, config.ConfigFileAbsolutePath(cfgFile))
	}

	if appSecret != "" {
		opts = append(opts, config.AppSecret(appSecret))
	}
	config.Init(opts...)

}

func main() {
	if err := RootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(-1)
	}
}
