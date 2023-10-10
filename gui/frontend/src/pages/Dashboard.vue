<template>
  <div>
    <div class="row">
      <div class="col-12">
        <card type="chart">
          <template slot="header">
            <div class="row">
              <div class="col-sm-6" :class="isRTL ? 'text-right' : 'text-left'">
                <h5 class="card-category">{{$t('dashboard.task')}}</h5>
                <h2 class="card-title"><i class="tim-icons icon-bell-55 text-primary "></i> {{$t('dashboard.performance')}}</h2>
              </div>
              <div class="col-sm-6">
                <div class="btn-group btn-group-toggle"
                     :class="isRTL ? 'float-left' : 'float-right'"
                     data-toggle="buttons">
                  <label v-for="(option, index) in bigLineChartCategories"
                         :key="option"
                         class="btn btn-sm btn-primary btn-simple"
                         :class="{active: bigLineChart.activeIndex === index}"
                         :id="index">
                    <input type="radio"
                           @click="initBigChart(index)"
                           name="options" autocomplete="off"
                           :checked="bigLineChart.activeIndex === index">
                    {{option}}
                  </label>
                </div>
              </div>
            </div>
          </template>
          <template>
            <div class="chart-area" style="height: max-content; margin-top: -2%;">
              <div class="image-group">
                <div class="image-item" :style="{ width: '50%', height: 'auto', float: 'left', padding:'2%', textAlign: 'center'}">
                  <img :src="bigLineChart.chartData.data[0]" alt="Acc"  />
                  <div class="image-caption" :style="{paddingTop:'2%'}">{{ bigLineChart.chartData.labels[0] }}</div>
                </div>
                <div class="image-item" :style="{ width: '50%', height: 'auto', float: 'right', padding:'2%', textAlign: 'center'}">
                  <img :src="bigLineChart.chartData.data[1]" alt="Acc"  />
                  <div class="image-caption" :style="{paddingTop:'2%'}">{{ bigLineChart.chartData.labels[1] }}</div>
                </div>
              </div>
            </div>
          </template>

          
        </card>
      </div>
    </div>
    <div class="row">
      <!-- <div class="col-lg-4" :class="{'text-right': isRTL}">
        <card type="chart">
          <template slot="header">
            <h5 class="card-category">{{$t('dashboard.task')}}</h5>
            <h3 class="card-title"><i class="tim-icons icon-bell-55 text-primary "></i> 763,215</h3>
          </template>
          <div class="chart-area">
            <line-chart style="height: 100%"
                        chart-id="purple-line-chart"
                        :chart-data="purpleLineChart.chartData"
                        :gradient-colors="purpleLineChart.gradientColors"
                        :gradient-stops="purpleLineChart.gradientStops"
                        :extra-options="purpleLineChart.extraOptions">
            </line-chart>
          </div>
        </card>
      </div> -->
      <div class="col-lg-4" :class="{'text-right': isRTL}">
        <card type="chart">
          <template slot="header">
            <h5 class="card-category">{{$t('dashboard.timeCost')}}</h5>
            <h3 class="card-title"><i class="tim-icons icon-delivery-fast text-info "></i> Time Cost (s)</h3>
          </template>
          <div class="chart-area">
            <bar-chart style="height: 100%"
                       chart-id="blue-bar-chart"
                       :chart-data="timeChart.chartData"
                       :gradient-stops="timeChart.gradientStops"
                       :extra-options="timeChart.extraOptions">
            </bar-chart>
          </div>
        </card>
      </div>
      <div class="col-lg-4" :class="{'text-right': isRTL}">
        <card type="chart">
          <template slot="header">
            <h5 class="card-category">{{$t('dashboard.train')}}</h5>
            <h3 class="card-title"><i class="tim-icons icon-send text-success "></i> Accuracy</h3>
          </template>
          <div class="chart-area">
            <bar-chart style="height: 100%"
                       chart-id="blue-bar-chart"
                       :chart-data="TrainAccChart.chartData"
                       :gradient-stops="TrainAccChart.gradientStops"
                       :extra-options="TrainAccChart.extraOptions">
            </bar-chart>
          </div>
        </card>
      </div>
      <div class="col-lg-4" :class="{'text-right': isRTL}">
        <card type="chart">
          <template slot="header">
            <h5 class="card-category">{{$t('dashboard.test')}}</h5>
            <h3 class="card-title"><i class="tim-icons icon-send text-success "></i> Accuracy</h3>
          </template>
          <div class="chart-area">
            <bar-chart style="height: 100%"
                       chart-id="blue-bar-chart"
                       :chart-data="TestAccChart.chartData"
                       :gradient-stops="TestAccChart.gradientStops"
                       :extra-options="TestAccChart.extraOptions">
            </bar-chart>
          </div>
        </card>
      </div>
      <!-- <div class="col-lg-4" :class="{'text-right': isRTL}">
        <card type="chart">
          <template slot="header">
            <h5 class="card-category">{{$t('dashboard.completedTasks')}}</h5>
            <h3 class="card-title"><i class="tim-icons icon-send text-success "></i> 12,100K</h3>
          </template>
          <div class="chart-area">
            <line-chart style="height: 100%"
                        chart-id="green-line-chart"
                        :chart-data="greenLineChart.chartData"
                        :gradient-stops="greenLineChart.gradientStops"
                        :extra-options="greenLineChart.extraOptions">
            </line-chart>
          </div>
        </card>
      </div> -->
    </div>
    <!-- <div class="row">
      <div class="col-lg-6 col-md-12">
        <card type="tasks" :header-classes="{'text-right': isRTL}">
          <template slot="header">
            <h6 class="title d-inline">{{$t('dashboard.tasks', {count: 5})}}</h6>
            <p class="card-category d-inline">{{$t('dashboard.today')}}</p>
            <base-dropdown menu-on-right=""
                           tag="div"
                           title-classes="btn btn-link btn-icon"
                           aria-label="Settings menu"
                           :class="{'float-left': isRTL}">
              <i slot="title" class="tim-icons icon-settings-gear-63"></i>
              <a class="dropdown-item" href="#pablo">{{$t('dashboard.dropdown.action')}}</a>
              <a class="dropdown-item" href="#pablo">{{$t('dashboard.dropdown.anotherAction')}}</a>
              <a class="dropdown-item" href="#pablo">{{$t('dashboard.dropdown.somethingElse')}}</a>
            </base-dropdown>
          </template>
          <div class="table-full-width table-responsive">
            <task-list></task-list>
          </div>
        </card>
      </div>
      <div class="col-lg-6 col-md-12">
        <card class="card" :header-classes="{'text-right': isRTL}">
          <h4 slot="header" class="card-title">{{$t('dashboard.simpleTable')}}</h4>
          <div class="table-responsive">
            <user-table></user-table>
          </div>
        </card>
      </div>
    </div> -->
  </div>
</template>
<script>
  // import LineChart from '@/components/Charts/LineChart';
  import BarChart from '@/components/Charts/BarChart';
  import * as chartConfigs from '@/components/Charts/config';
  // import TaskList from './Dashboard/TaskList';
  // import UserTable from './Dashboard/UserTable';
  import config from '@/config';

  export default {
    components: {
      // LineChart,
      BarChart
      // TaskList,
      // UserTable
    },
    data() {
      return {
        bigLineChart: {
          allData: [
            [
              require('@/assets/pic/Accuracy_ResNet34.png'),
              require('@/assets/pic/Loss_ResNet34.png')
            ],
            [
            require('@/assets/pic/Accuracy_ViT.png'),
              require('@/assets/pic/Loss_ViT.png')
            ]
          ],
          activeIndex: 0,
          chartData: {
            data: [],
            labels: ['Accuracy', 'Loss'],
          },
          // extraOptions: chartConfigs.purpleChartOptions,
          // gradientColors: config.colors.primaryGradient,
          // gradientStops: [1, 0.4, 0],
          // categories: []
        },
        // purpleLineChart: {
        //   extraOptions: chartConfigs.purpleChartOptions,
        //   chartData: {
        //     labels: ['JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'],
        //     datasets: [{
        //       label: "Data",
        //       fill: true,
        //       borderColor: config.colors.primary,
        //       borderWidth: 2,
        //       borderDash: [],
        //       borderDashOffset: 0.0,
        //       pointBackgroundColor: config.colors.primary,
        //       pointBorderColor: 'rgba(255,255,255,0)',
        //       pointHoverBackgroundColor: config.colors.primary,
        //       pointBorderWidth: 20,
        //       pointHoverRadius: 4,
        //       pointHoverBorderWidth: 15,
        //       pointRadius: 4,
        //       data: [80, 100, 70, 80, 120, 80],
        //     }]
        //   },
        //   gradientColors: config.colors.primaryGradient,
        //   gradientStops: [1, 0.2, 0],
        // },
        // greenLineChart: {
        //   extraOptions: chartConfigs.greenChartOptions,
        //   chartData: {
        //     labels: ['JUL', 'AUG', 'SEP', 'OCT', 'NOV'],
        //     datasets: [{
        //       label: "My First dataset",
        //       fill: true,
        //       borderColor: config.colors.danger,
        //       borderWidth: 2,
        //       borderDash: [],
        //       borderDashOffset: 0.0,
        //       pointBackgroundColor: config.colors.danger,
        //       pointBorderColor: 'rgba(255,255,255,0)',
        //       pointHoverBackgroundColor: config.colors.danger,
        //       pointBorderWidth: 20,
        //       pointHoverRadius: 4,
        //       pointHoverBorderWidth: 15,
        //       pointRadius: 4,
        //       data: [90, 27, 60, 12, 80],
        //     }]
        //   },
        //   gradientColors: ['rgba(66,134,121,0.15)', 'rgba(66,134,121,0.0)', 'rgba(66,134,121,0)'],
        //   gradientStops: [1, 0.4, 0],
        // },
        timeChart: {
          extraOptions: chartConfigs.barChartOptions,
          chartData: {
            labels: ['ResNet', 'ViT'],
            datasets: [{
              label: "time(s)",
              fill: true,
              borderColor: config.colors.info,
              borderWidth: 2,
              borderDash: [],
              borderDashOffset: 0.0,
              data: [8.34, 159.7],
            }]
          },
          gradientColors: config.colors.primaryGradient,
          gradientStops: [1, 0.4, 0]
        },
        TrainAccChart: {
          extraOptions: chartConfigs.barChartOptions,
          chartData: {
            labels: ['ResNet', 'ViT'],
            datasets: [{
              label: "Acc",
              fill: true,
              borderColor: config.colors.info,
              borderWidth: 2,
              borderDash: [],
              borderDashOffset: 0.0,
              data: [1, 0.929],
            }]
          },
          gradientColors: config.colors.primaryGradient,
          gradientStops: [1, 0.4, 0]
        },
        TestAccChart: {
          extraOptions: chartConfigs.barChartOptions,
          chartData: {
            labels: ['ResNet', 'ViT'],
            datasets: [{
              label: "Acc",
              fill: true,
              borderColor: config.colors.info,
              borderWidth: 2,
              borderDash: [],
              borderDashOffset: 0.0,
              data: [0.93, 0.886],
            }]
          },
          gradientColors: config.colors.primaryGradient,
          gradientStops: [1, 0.4, 0]
        }
      }
    },
    computed: {
      enableRTL() {
        return this.$route.query.enableRTL;
      },
      isRTL() {
        return this.$rtl.isRTL;
      },
      bigLineChartCategories() {
        return this.$t('dashboard.chartCategories');
      }
    },
    methods: {
      initBigChart(index) {
        let chartData = {
          data: this.bigLineChart.allData[index],
          labels: ['Accuracy', 'Loss']
        }
        this.bigLineChart.chartData = chartData;
        this.bigLineChart.activeIndex = index;
      }
    },
    mounted() {
      this.i18n = this.$i18n;
      if (this.enableRTL) {
        this.i18n.locale = 'ar';
        this.$rtl.enableRTL();
      }
      this.initBigChart(0);
    },
    beforeDestroy() {
      if (this.$rtl.isRTL) {
        this.i18n.locale = 'en';
        this.$rtl.disableRTL();
      }
    }
  };
</script>
<style>
</style>
