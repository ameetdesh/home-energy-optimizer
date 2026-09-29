package main

import (
	"context"
	"fmt"
	"os"

	optimizer "github.com/evcc-io/optimizer/client"
)

func f32s(n int, fn func(int) float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = fn(i)
	}
	return out
}

func main() {
	c, err := optimizer.NewClientWithResponses(os.Args[1])
	if err != nil {
		panic(err)
	}
	ctx := context.Background()

	h, err := c.GetOptimizeHealthWithResponse(ctx)
	if err != nil {
		panic(err)
	}
	fmt.Printf("health: HTTP %d status=%q message=%q\n",
		h.HTTPResponse.StatusCode, h.JSON200.Status, h.JSON200.Message)

	n := 96
	dt := make([]int, n)
	for i := range dt {
		dt[i] = 900
	}
	dt[0] = 420 // evcc's short first slot

	req := optimizer.OptimizationInput{
		Batteries: []optimizer.BatteryConfig{{
			CMax: 5000, CMin: 0, DMax: 5000,
			SInitial: 5000, SMin: 0, SMax: 10000, SCapacity: 10000,
			PA: 0.20 / 1000,
		}},
		EtaC: 0.95, EtaD: 0.95,
		Grid: optimizer.GridConfig{PMaxImp: 17250, PMaxExp: 10000},
		Strategy: optimizer.OptimizerStrategy{
			ChargingStrategy: optimizer.OptimizerStrategyChargingStrategyChargeBeforeExport,
		},
		TimeSeries: optimizer.TimeSeries{
			Dt: dt,
			Gt: f32s(n, func(i int) float32 { return 250 }),
			Ft: f32s(n, func(i int) float32 {
				h := float64(i) * 0.25
				if h > 6 && h < 18 {
					return 800
				}
				return 0
			}),
			PN: f32s(n, func(i int) float32 {
				if float64(i)*0.25 < 7 {
					return 0.20 / 1000
				}
				return 0.60 / 1000
			}),
			PE: f32s(n, func(i int) float32 { return 0.05 / 1000 }),
		},
	}

	resp, err := c.PostOptimizeChargeScheduleWithResponse(ctx, req)
	if err != nil {
		panic(err)
	}
	if resp.JSON200 == nil {
		fmt.Printf("FAILED HTTP %d body=%s\n", resp.HTTPResponse.StatusCode, string(resp.Body))
		os.Exit(1)
	}
	r := resp.JSON200
	fmt.Printf("schedule: HTTP %d status=%q objective=%.4f\n",
		resp.HTTPResponse.StatusCode, r.Status, r.ObjectiveValue)
	fmt.Printf("  batteries=%d charging=%d discharging=%d soc=%d\n",
		len(r.Batteries), len(r.Batteries[0].ChargingPower),
		len(r.Batteries[0].DischargingPower), len(r.Batteries[0].StateOfCharge))
	fmt.Printf("  flow=%d gridImport=%d gridExport=%d\n",
		len(r.FlowDirection), len(r.GridImport), len(r.GridExport))
	var chg, dis float32
	for i := range r.Batteries[0].ChargingPower {
		chg += r.Batteries[0].ChargingPower[i]
		dis += r.Batteries[0].DischargingPower[i]
	}
	fmt.Printf("  total charge=%.0f Wh discharge=%.0f Wh soc[0]=%.0f soc[last]=%.0f\n",
		chg, dis, r.Batteries[0].StateOfCharge[0],
		r.Batteries[0].StateOfCharge[len(r.Batteries[0].StateOfCharge)-1])
	fmt.Printf("  limitViolations: import=%v exportHit=%v\n",
		r.LimitViolations.GridImportLimitExceeded, r.LimitViolations.GridExportLimitHit)
	fmt.Println("OK: evcc's generated client parsed the response")
}
