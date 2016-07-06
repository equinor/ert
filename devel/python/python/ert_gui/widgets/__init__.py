#  Copyright (C) 2011  Statoil ASA, Norway. 
#   
#  The file '__init__.py' is part of ERT - Ensemble based Reservoir Tool. 
#   
#  ERT is free software: you can redistribute it and/or modify 
#  it under the terms of the GNU General Public License as published by 
#  the Free Software Foundation, either version 3 of the License, or 
#  (at your option) any later version. 
#   
#  ERT is distributed in the hope that it will be useful, but WITHOUT ANY 
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or 
#  FITNESS FOR A PARTICULAR PURPOSE.   
#   
#  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html> 
#  for more details. 


def addHelpToWidget(widget, link):
    original_enter_event = widget.enterEvent

    def enterEvent(event):
        original_enter_event(event)
        try:
            from ert_gui.tools import HelpCenter
            HelpCenter.getHelpCenter("ERT").setHelpMessageLink(link)
        except AttributeError:
            pass

    widget.enterEvent = enterEvent