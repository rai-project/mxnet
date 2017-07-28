package main

import (
	"fmt"
	"net"
	"os"
	"path/filepath"

	"github.com/Unknwon/com"
	"github.com/fatih/color"
	"github.com/pkg/errors"
	"github.com/rai-project/config"
	"github.com/rai-project/mxnet/agent"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var (
	isDebug   bool
	isVerbose bool
	appSecret string
	cfgFile   string
)

const (
	DEFAULTPORT = "9999"
)

func getAddress(port string) string {
	return "localhost:" + port
}

// RootCmd represents the base command when called without any subcommands
var RootCmd = &cobra.Command{
	Use:   "mxnet-agent",
	Short: "A brief description of your application",
	Long: `A longer description that spans multiple lines and likely contains
examples and usage of using your application. For example:

Cobra is a CLI library for Go that empowers applications.
This application is a tool to generate the needed files
to quickly create a Cobra application.`,
	RunE: func(cmd *cobra.Command, args []string) error {
		port, found := os.LookupEnv("PORT")
		if !found {
			port = DEFAULTPORT
		}
		server := agent.Register()

		address := getAddress(port)
		lis, err := net.Listen("tcp", address)
		if err != nil {
			return errors.Wrapf(err, "failed to listen on ip %s", address)
		}

		log.Infof("mxnet service is listening on %s", address)

		server.Serve(lis)
		server.GracefulStop()
		return nil
	},
}

func init() {
	cobra.OnInitialize(initConfig)

	// Here you will define your flags and configuration settings.
	// Cobra supports Persistent Flags, which, if defined here,
	// will be global for your application.

	RootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", "config file (default is $HOME/.mxnet-agent.yaml)")
	RootCmd.PersistentFlags().BoolVarP(&isVerbose, "verbose", "v", false, "Toggle verbose mode.")
	RootCmd.PersistentFlags().BoolVarP(&isDebug, "debug", "d", false, "Toggle debug mode.")
	RootCmd.PersistentFlags().StringVarP(&appSecret, "secret", "s", "", "The application secret.")

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
		config.AppName("mxnet-agent"),
		config.ColorMode(true),
	}
	if com.IsFile(cfgFile) {
		if c, err := filepath.Abs(cfgFile); err == nil {
			cfgFile = c
		}
		opts = append(opts, config.ConfigFileAbsolutePath(cfgFile))
	} else {
		opts = append(opts, config.ConfigString(""))
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
