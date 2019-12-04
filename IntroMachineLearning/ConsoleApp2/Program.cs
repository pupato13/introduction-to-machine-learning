using Accord.Controls;
using Accord.Math.Optimization.Losses;
using Accord.Statistics.Models.Regression.Linear;
using Accord.Statistics.Visualizations;
using Deedle;
using System;
using System.IO;
using System.Linq;


namespace ConsoleApp2
{
    class Program
    {
        static readonly string _dataPath
            = Path.Combine(Path.GetDirectoryName(Path.GetDirectoryName(Path.GetDirectoryName(AppDomain.CurrentDomain.BaseDirectory))),
                "Data",
                "california_housing.csv");

        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            Console.WriteLine("Folder: " + _dataPath);

            // This will load the file in memory in a Frame instance.
            var housing = Frame.ReadCsv(_dataPath, separators: ",");

            housing = housing.Where(kv => ((decimal)kv.Value["median_house_value"]) < 500000);

            // set up a few series
            var total_rooms = housing["total_rooms"];
            var median_house_value = housing["median_house_value"];
            var median_income = housing["median_income"];

            // The median_house_value column is in the range from 0 to 500,000. To make these values more manageable, let's divide all values by 1,000:
            median_house_value /= 1000;

            RunningSimpleLinearRegression(total_rooms, median_house_value, median_income);

            Console.ReadLine();
        }

        /// <summary>
        /// We are going to assume that there is a linear relationship between the total number of rooms in a housing block and the median house value of that same block.
        /// To test our hypothesis, we're going to run a linear regression on the data and see if we get a good fit.
        /// We're going to use the machine learning classes in the Accord.NET library. You can find them in the Accord.MachineLearning package.
        /// The Accord regression classes expect data in the form of array of double. So we need to convert our Deedle series into double arrays.
        /// The ValuesAll property is exactly what we need; it returns all values in the series as an enumeration.So we get the following code:
        /// </summary>
        /// <param name="total_rooms"></param>
        /// <param name="median_house_value"></param>
        private static void RunningSimpleLinearRegression(Series<int, double> total_rooms, Series<int, double> median_house_value, Series<int, double> median_income)
        {
            // set up feature and label
            // This gets us both the input features (total_rooms) and the output labels (median_house_value) as arrays of double.
            //var feature = total_rooms.Values.ToArray();
            var feature = median_income.Values.ToArray();
            var labels = median_house_value.Values.ToArray();



            // The next step is to pick the learning algorithm.
            // We could use gradient descent, but since we're doing linear regression with only a single input feature, there is an even better solution
            // that will give us the perfect fit in just a single pass: the ### OrdinaryLeastSquares ### class. Here's how that works:

            // train the model
            // This code snippet will run a linear regression on the data, using the ordinary least squares algorithm to find the optimal solution.
            var learner = new OrdinaryLeastSquares();
            var model = learner.Learn(feature, labels);

            // We can access the discovered model parameters by reading the Slope and Intercept properties, like this:
            Console.WriteLine($"Slope:      {model.Slope}");
            Console.WriteLine($"Intercept:  {model.Intercept}");

            // #####################
            // Validating The Result
            // #####################

            // So is this a good fit? To find out, we must validate the model. We can do this by running every single feature through the model; this will yield a set of predictions.
            // Then we can compare each prediction with the actual label, and calculate the Root Mean Squared Error (RMSE) value:

            // validate the model
            var predictions = model.Transform(feature);
            var rmse = Math.Sqrt(new SquareLoss(labels).Loss(predictions));

            // The RMSE indicates the uncertainty in each prediction.
            // We can compare it to the range of labels to get a feel for the accuracy of the model:

            var range = Math.Abs(labels.Max() - labels.Min());
            Console.WriteLine($"Label range:    {range}");
            Console.WriteLine($"RMSE:           {rmse} {rmse / range * 100:0.00}%");

            // RESULTS
            // Slope:       0.006969381760507163
            // Intercept:   188.8762058206879
            // Label range: 485.00199999999995
            // RMSE:        114.98100785209695 23.71%

            // We get an RMSE of 114, which is more than 23% of the label range. That's not very good.


            // Let's plot the data and the regression line to get a better feel for the data.
            // Accord.NET has a built-in graph library for quickly creating scatterplots and histograms.
            // To use it, you first need to install the Accord.Controls Nuget package.

            // Now we need to get a little creative.
            // Accord can work with separate x- and y data arrays (corresponding nicely to our feature and labels variables),
            // but we need to plot two data series: the labels and the model predictions.
            // To get this to work, we need to concatenate the labels and predictions arrays together.
            // The following code sets up two x- and y value arrays for the plot:

            // generate plot arrays
            var x = feature.Concat(feature).ToArray();
            var y = predictions.Concat(labels).ToArray();

            // Finally, we need a third array to tell Accord what color to use when drawing the two series.
            // We will generate an array with the value 1 for all predictions, and 2 for all labels:

            // set up color array
            var colors1 = Enumerable.Repeat(1, labels.Length).ToArray();
            var colors2 = Enumerable.Repeat(2, labels.Length).ToArray();
            var c = colors1.Concat(colors2).ToArray();

            // And now we can generate the scatterplot:

            // plot the data
            var plot = new Scatterplot("Training", "feature", "label");
            plot.Compute(x, y, c);
            ScatterplotBox.Show(plot);
        }
    }
}
