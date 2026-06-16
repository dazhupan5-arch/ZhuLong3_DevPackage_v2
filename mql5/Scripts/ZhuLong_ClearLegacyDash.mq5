//+------------------------------------------------------------------+
//|                                    ZhuLong_ClearLegacyDash.mq5 |
//|  清除烛龙智核PRO / 旧版仪表盘残留（TS30_* / TSFB_* 图表对象）   |
//|  不影响烛龙3 ZhuLongIndicator（ZL_* 信号对象保留）                |
//+------------------------------------------------------------------+
#property copyright "烛龙 ZhuLong"
#property version   "1.00"
#property script_show_inputs

input bool InpAlsoRemoveV30Indicator = true;  // 同时从本图卸载智核仪表盘指标

int DeleteByPrefixes(const long chartId, const string &prefixes[])
{
   int deleted = 0;
   const int total = ObjectsTotal(chartId, 0, -1);
   for(int i = total - 1; i >= 0; i--)
   {
      string name = ObjectName(chartId, i, 0, -1);
      if(StringLen(name) < 4)
         continue;
      bool hit = false;
      for(int p = 0; p < ArraySize(prefixes); p++)
      {
         if(StringFind(name, prefixes[p], 0) == 0)
         {
            hit = true;
            break;
         }
      }
      if(!hit)
         continue;
      if(ObjectDelete(chartId, name))
         deleted++;
   }
   return deleted;
}

void RemoveLegacyIndicators(const long chartId)
{
   const int n = ChartIndicatorsTotal(chartId, 0);
   for(int i = n - 1; i >= 0; i--)
   {
      string name = ChartIndicatorName(chartId, 0, i);
      if(StringFind(name, "智核仪表盘") >= 0 ||
         StringFind(name, "智核PRO") >= 0 ||
         StringFind(name, "TSFB") >= 0)
      {
         ChartIndicatorDelete(chartId, 0, name);
         Print("已卸载指标: ", name);
      }
   }
}

void OnStart()
{
   const long cid = ChartID();
   string prefixes[];
   ArrayResize(prefixes, 2);
   prefixes[0] = "TS30_";
   prefixes[1] = "TSFB_";

   int n = DeleteByPrefixes(cid, prefixes);
   if(InpAlsoRemoveV30Indicator)
      RemoveLegacyIndicators(cid);

   ChartRedraw(cid);
   Print("烛龙清理完成：删除 ", n, " 个残留对象（前缀 TS30_/TSFB_）");
   Alert("烛龙清理完成：删除 ", n, " 个残留对象");
}
