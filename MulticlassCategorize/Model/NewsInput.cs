using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MulticlassCategorize.Model
{
    class NewsInput
    {
        [LoadColumn(0)]
        public string Label { get; set; }
        [LoadColumn(1)]
        public string Title { get; set; }
        [LoadColumn(2)]
        public string Body { get; set; }
    }
}
